"""
build_index.py — Build BM25 and FAISS retrieval indexes over Wikipedia passages.

Tier A: FEVER wiki-pages (2017 Wikipedia intros, gold evidence labels).
         Download from fever.ai:
           wget https://s3-eu-west-1.amazonaws.com/fever.ai/wiki-pages.zip
           unzip wiki-pages.zip -d data/raw/fever/

Tier B: Scoped subset of Wikipedia 20231101.en (Phase 5+).

Usage:
    python scripts/build_index.py --tier A --output_dir indexes/ --encoder BAAI/bge-small-en-v1.5

Colab:
    !python scripts/build_index.py \\
        --tier A \\
        --output_dir /content/drive/MyDrive/hallucination_project/indexes
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Text chunking ────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 20) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ─── Tier A: FEVER wiki-pages ─────────────────────────────────────────────────

def build_passage_list_tier_a(
    output_dir: Path,
    chunk_size: int = 150,
    chunk_overlap: int = 20,
    max_passages: int = 1_500_000,
) -> list[dict]:
    """
    Build passage list from FEVER wiki-pages.
    Expects data/raw/fever/wiki-pages/*.jsonl
    """
    wiki_dir = Path("data/raw/fever/wiki-pages")
    if not wiki_dir.exists():
        print("⚠  FEVER wiki-pages not found.")
        print("   Download: wget https://s3-eu-west-1.amazonaws.com/fever.ai/wiki-pages.zip")
        print("   Extract:  unzip wiki-pages.zip -d data/raw/fever/")
        return []

    passages: list[dict] = []
    jsonl_files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    print(f"  Found {len(jsonl_files)} FEVER wiki-page files")

    for jsonl_file in jsonl_files:
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                title = doc.get("id", "Unknown")
                raw_lines = doc.get("lines", "")
                # FEVER format: "0\tsentence\n1\tsentence\n..."
                text_parts = []
                for ln in raw_lines.split("\n"):
                    parts = ln.split("\t", 1)
                    if len(parts) == 2 and parts[1].strip():
                        text_parts.append(parts[1].strip())
                text = " ".join(text_parts)
                if not text.strip():
                    continue

                for chunk_id, chunk in enumerate(
                    chunk_text(text, chunk_size, chunk_overlap)
                ):
                    passages.append({
                        "doc_id": f"{title}_{chunk_id}",
                        "title": title,
                        "chunk_id": chunk_id,
                        "text": f"Title: {title}. {chunk}",
                    })
                    if len(passages) >= max_passages:
                        break
            if len(passages) >= max_passages:
                break

    print(f"  Tier A: {len(passages):,} passages")

    # Save passage list
    output_dir.mkdir(parents=True, exist_ok=True)
    passages_path = output_dir / "passages_A.jsonl"
    with open(passages_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    print(f"  Saved passages → {passages_path}")
    return passages


# ─── Tier B: scoped Wikipedia (Phase 5+) ─────────────────────────────────────

def build_passage_list_tier_b(
    output_dir: Path,
    entity_titles: list[str] | None = None,
    n_distractor_articles: int = 300_000,
    chunk_size: int = 150,
    chunk_overlap: int = 20,
    first_n_words: int = 500,
    max_passages: int = 1_500_000,
) -> list[dict]:
    """
    Build scoped Wikipedia subset using streaming (Colab-friendly).
    Keeps: (a) articles whose title matches entity_titles, (b) random distractors.
    """
    from datasets import load_dataset

    print("  Streaming Wikipedia 20231101.en (this may take a while)…")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                      streaming=True, trust_remote_code=True)

    entity_set = set(t.lower() for t in (entity_titles or []))
    passages: list[dict] = []
    kept = 0
    distractor_budget = n_distractor_articles
    import random as rnd

    for article in ds:
        title = article.get("title", "")
        text = article.get("text", "")
        # Truncate to first_n_words
        words = text.split()[:first_n_words]
        text = " ".join(words)
        if not text.strip():
            continue

        keep = title.lower() in entity_set
        if not keep and distractor_budget > 0:
            if rnd.random() < 0.5:
                keep = True
                distractor_budget -= 1

        if keep:
            for chunk_id, chunk in enumerate(
                chunk_text(text, chunk_size, chunk_overlap)
            ):
                passages.append({
                    "doc_id": f"{title}_{chunk_id}",
                    "title": title,
                    "chunk_id": chunk_id,
                    "text": f"Title: {title}. {chunk}",
                })
            kept += 1

        if len(passages) >= max_passages:
            break

    print(f"  Tier B: {kept:,} articles → {len(passages):,} passages")
    output_dir.mkdir(parents=True, exist_ok=True)
    passages_path = output_dir / "passages_B.jsonl"
    with open(passages_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    print(f"  Saved passages → {passages_path}")
    return passages


# ─── BM25 index ───────────────────────────────────────────────────────────────

def build_bm25_index(
    passages: list[dict],
    output_dir: Path,
    tier: str,
) -> None:
    from rank_bm25 import BM25Okapi

    print(f"\n  Building BM25 index over {len(passages):,} passages…")
    tokenized = [p["text"].lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized)
    index_path = output_dir / f"bm25_{tier}.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "passages": passages}, f)
    print(f"  BM25 index saved → {index_path}")


# ─── FAISS index ──────────────────────────────────────────────────────────────

def build_faiss_index(
    passages: list[dict],
    output_dir: Path,
    tier: str,
    encoder_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256,
    shard_size: int = 50_000,
) -> None:
    """
    Encode passages in shards (Colab-safe: resumes if session disconnects).
    Each shard is saved as a .npy file; existing shards are skipped.
    """
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch

    shard_dir = output_dir / f"faiss_shards_{tier}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading encoder {encoder_name}…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(encoder_name, device=device)
    model.half()  # fp16 for memory efficiency

    all_embeddings: list[np.ndarray] = []

    for shard_start in range(0, len(passages), shard_size):
        shard_path = shard_dir / f"shard_{shard_start:08d}.npy"
        if shard_path.exists():
            print(f"  Shard {shard_start}: loading cached ✓")
            embs = np.load(shard_path)
        else:
            batch = passages[shard_start: shard_start + shard_size]
            texts = [p["text"] for p in batch]
            print(f"  Encoding shard {shard_start}–{shard_start + len(batch) - 1}…")
            embs = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            np.save(shard_path, embs)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype("float32")
    dim = embeddings.shape[1]
    n = len(passages)
    print(f"\n  Building FAISS index: {n:,} vectors, dim={dim}")

    if n > 1_000_000:
        nlist = min(4096, n // 100)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = 64
    else:
        index = faiss.IndexFlatIP(dim)  # exact search ≤1M

    index.add(embeddings)
    faiss_path = output_dir / f"faiss_{tier}.bin"
    faiss.write_index(index, str(faiss_path))
    print(f"  FAISS index saved → {faiss_path}  ({index.ntotal:,} vectors)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BM25 and FAISS retrieval indexes."
    )
    parser.add_argument("--tier", choices=["A", "B"], default="A",
                        help="A = FEVER wiki-pages; B = scoped Wikipedia")
    parser.add_argument("--output_dir", default="indexes",
                        help="Directory to save index files")
    parser.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--chunk_size", type=int, default=150)
    parser.add_argument("--chunk_overlap", type=int, default=20)
    parser.add_argument("--max_passages", type=int, default=1_500_000)
    parser.add_argument("--bm25_only", action="store_true",
                        help="Skip FAISS index (faster for dev)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.tier == "A":
        passages = build_passage_list_tier_a(
            output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_passages=args.max_passages,
        )
    else:
        passages = build_passage_list_tier_b(
            output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_passages=args.max_passages,
        )

    if not passages:
        print("No passages built. Exiting.")
        return

    build_bm25_index(passages, output_dir, args.tier)

    if not args.bm25_only:
        build_faiss_index(passages, output_dir, args.tier, encoder_name=args.encoder)

    print("\n✅ Index build complete.")
    print("Next step: run notebooks/03_run_pipeline.ipynb")


if __name__ == "__main__":
    main()
