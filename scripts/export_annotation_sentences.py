"""Export human-readable sentence offsets for a prepared blank annotation batch."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decompose import split_sentences_with_spans
from src.experiments.artifacts import canonical, file_hash, load_inputs
from src.experiments.protocol import require_segmenter, validate_config
from src.experiments.runner import pinned_segmenter


def export_sentences(batch_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    batch = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    config = validate_config(batch["config"])
    nlp = require_segmenter(config)
    rows = {}
    for split in ("dev", "test"):
        relative = f"inputs/{split}.jsonl"
        source = batch_dir / relative
        if file_hash(source) != batch["artifact_sha256"][relative]:
            raise ValueError("Prepared inputs changed; create a new batch version")
        records = load_inputs(source, split)
        if {record.id for record in records} != set(batch["split_ids"][f"{split}_ids"]):
            raise ValueError("Prepared split IDs differ from inputs")
        with pinned_segmenter(nlp):
            rows[split] = [{"id": record.id, "question": record.question, "answer": record.answer,
                            "sentences": [{"sentence_id": i, "text": text, "span": [start, end]}
                                          for i, (text, start, end) in enumerate(split_sentences_with_spans(record.answer))]}
                           for record in records]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sentences-", dir=output_dir.parent))
    try:
        for split, records in rows.items():
            (staging / f"{split}.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in records))
            text = []
            for row in records:
                text += [f"ID: {row['id']}", f"Question: {row['question']}", f"Original answer: {row['answer']}"]
                text += [f"  sentence_id={item['sentence_id']} span={item['span']}: {item['text']}" for item in row["sentences"]]
                text.append("")
            (staging / f"{split}.txt").write_text("\n".join(text), encoding="utf-8")
        manifest = {"schema_version": 1, "status": "sentence_reference_only_not_gold",
                    "batch_manifest_sha256": file_hash(batch_dir / "manifest.json"),
                    "segmenter": config["segmentation"], "spacy_version": version("spacy"),
                    "pilot_dev_ids": [row["id"] for row in rows["dev"][:5]],
                    "files_sha256": {path.name: file_hash(path) for path in sorted(staging.iterdir())}}
        (staging / "manifest.json").write_bytes(canonical(manifest) + b"\n")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = export_sentences(args.batch_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Sentence export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Sentence reference saved: {output}; human claim extraction and labels remain required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
