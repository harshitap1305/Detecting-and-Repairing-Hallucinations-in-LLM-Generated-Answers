# Path A + C training workflow

The [Colab notebook](../notebooks/04_train_A_C_colab.ipynb) prepares and trains two additions to the existing pipeline:

- **A — verifier:** a three-class DeBERTa model trained on FEVER claim/evidence pairs. It selects a checkpoint using an article-disjoint internal development split, then evaluates on an article-disjoint subset of the official labelled FEVER development file.
- **C — repair:** a Flan-T5 editor trained on deterministic corruptions of Wikipedia sentences. A candidate enters training only after a verifier labels the original `Supported` and the corruption `Contradicted` or `Unverifiable`. The train/dev/test split is by Wikipedia article.

The current repository does not include the FEVER files, trained checkpoints, course-project gold labels, or per-record baseline predictions. The notebook creates the data and checkpoints in **your own Colab/Drive session**; no benchmark result should be reported until its commands finish. It starts with a bounded pilot. Change `PILOT = False` only after the pilot works and you have a project holdout manifest and enough Colab storage/RAM. Colab GPU availability and session limits vary; choose a GPU runtime before running the notebook.

## Inputs and persistent outputs

The notebook downloads the [official FEVER training data, labelled shared-task development data, and Wikipedia pages](https://fever.ai/dataset/fever.html) into `MyDrive/hallucination_project_ac/fever`. The Wikipedia archive, extracted pages, and a large BM25 index need substantial disk and RAM. The notebook stores retrieval JSONL, split manifests, synthetic candidates/verdicts, model checkpoints, metrics, the exact Git commit, installed package versions, and input hashes under the same Drive folder. It extracts the wiki files and copies the index into `/content` for faster processing during each session.

`requirements-training.txt` installs the model-training, BM25 retrieval, and Groq dependencies. The optional end-to-end cell installs spaCy and its English model for consistent sentence segmentation. The notebook checks that CUDA is available before the training cells. GPU type is recorded, not assumed. FEVER inputs are downloaded to a `.part` file and renamed after a successful download, so an interrupted download is not mistaken for a complete dataset.

For the final course-project comparison, provide a private JSON holdout manifest with lists named `article_ids`, `entities`, `prompt_ids`, and `prompts`, plus annotated `data/gold/dev.jsonl`, `data/gold/test.jsonl`, and a populated `data/splits.json`. Include at least one course-test article ID or entity; prompt IDs and prompt text alone cannot exclude the Wikipedia source sentences used for Path C. Use the course-test entities and prompts in that manifest **before** building repair candidates or training the verifier. `PILOT = False` refuses to start without the manifest and an article/entity key. The pilot reserves articles named in official FEVER development evidence, but that alone does not protect an unprovided course-project test set.

## Run order

1. Open `notebooks/04_train_A_C_colab.ipynb` in Colab, select a GPU runtime, mount Drive, and run cells in order. The setup cell clones `codex/train-verifier-repair`; change `REPO_REF` only if the work is merged into another branch.
2. The notebook downloads FEVER, extracts wiki pages, builds one shared BM25 index, and runs **label-independent** retrieval for train and official-dev claims. The pilot caps passages, claims, and repair source sentences. A capped index is a pilot retrieval environment, so its FEVER results are not comparable to a full-index benchmark.
3. It creates `verifier/train.jsonl` and `verifier/dev.jsonl`, fine-tunes DeBERTa, filters official FEVER development claims against training/development article and entity keys, and evaluates the selected model on the surviving held-out subset. The `verifier/manifest.json` and `verifier_holdout/manifest.json` record counts and skips; the internal development metrics are for checkpoint selection only.
4. It creates repair candidates, runs the existing NLI verifier to audit each candidate, creates article-disjoint `repair_train.jsonl`, `repair_dev.jsonl`, and `repair_test.jsonl`, trains Flan-T5, and evaluates the best checkpoint on the synthetic test split. The verdict JSONL is append/resume capable. Review its labels and a sample of accepted edits before treating the corpus as high-quality training data.
5. After course-project gold data is available, the optional final cell compares the prompted baseline, trained verifier only, trained repair only, and A+C on **the same held-out records**. It requires a Groq key in Colab Secrets and explicit gold/dev/test/index/checkpoint paths. The comparison script fails on missing inputs or incomplete outputs; it does not substitute published numbers.

All commands are visible in the notebook and can be run from the repository root. A direct final comparison command is:

```bash
python scripts/compare_training_paths.py \
  --inputs data/gold/test.jsonl \
  --gold-dev data/gold/dev.jsonl \
  --gold-test data/gold/test.jsonl \
  --splits data/splits.json \
  --verifier-checkpoint /path/to/verifier_run/final \
  --repair-checkpoint /path/to/repair_run/checkpoint-epoch-N \
  --index /path/to/bm25_A.pkl \
  --output-dir /path/to/ac_comparison
```

The comparison requires an actual LLM provider key for the baseline and decomposition. Keep it in Colab Secrets, never in a notebook output or commit. The prompted baseline and all three trained variants share the index and input IDs. Run `python scripts/audit_baseline.py` to see which repository baseline artifacts are still absent.

## Interpretation and limits

- `verifier_holdout` is a **filtered subset** of official FEVER development data; its count and class mix must be reported. It is not the complete FEVER shared-task score.
- Repair test exact match and token F1 measure editing on held-out **synthetic corruptions**. The generator uses the source sentence in its evidence, so these results do not establish repair quality on independently retrieved evidence or real LLM hallucinations. Use the course-project gold test and human review for that claim.
- FEVER evidence sentences are gold for `SUPPORTS`/`REFUTES`, while `NOT ENOUGH INFO` uses retrieved text. The prepared data records skipped claims; retrieval misses are not silently relabelled.
- The current BM25 index builder scans Wikipedia files in order and can cap passages. Increasing the cap increases memory use and still requires reporting the actual indexed passage count. A full claim run with the repository's naive BM25 retrieval can be slow; for large runs supply label-independent retrieval JSONL produced by a scalable retriever, then use the same preparation and evaluation scripts.
- Record exact data hashes, Git commit, package versions, seed, hardware, and checkpoint path alongside every reported number. Keep internal development metrics separate from held-out results, and do not claim a benefit of A+C before the paired course-project comparison completes.

The dataset format and files are described by [FEVER](https://fever.ai/dataset/fever.html). Runtime and Drive constraints are described by the [Colab FAQ](https://research.google.com/colaboratory/faq.html).
