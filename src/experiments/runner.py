"""Serial benchmark execution with immutable attempts and identity-checked resume."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import uuid

from scripts.audit_baseline import read_splits
from src.experiments import adapters
from src.experiments.artifacts import (IDENTITY, atomic_json, canonical, file_hash,
                                      fingerprint, load_inputs, read_inventory, validate_result)
from src.experiments.protocol import load_config, require_segmenter
from src.experiments.runtime import execution
from src.llm import endpoint_identity

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {"provider": "groq", "model": "llama-3.3-70b-versatile", "model_revision": "hosted-unpinned",
            "verifier": "llm", "top_k": 5, "max_repair_iterations": 2,
            "repair_policy": "rewrite+hedge", "n_samples": 5, "consistency_threshold": 0.5,
            "faiss_index_path": None, "passages_path": None,
            "query_strategy": "contextual_v1", "retrieval_max_queries": 3,
            "retrieval_max_passages": 20, "tracking_max_claims": 64, "repair_guard": "grounded_v1",
            "reasoning_effort": "low", "gptoss_min_completion_tokens": 4096}


def now():
    return datetime.now(timezone.utc).isoformat()


def code_identity() -> dict:
    from src.experiments.provenance import source_hashes
    hashes = source_hashes(ROOT)
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            raise ValueError("Cannot determine Git revision for provenance")
        return result.stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "source_sha256": fingerprint(hashes)}


def environment() -> dict:
    versions = {}
    for name in ("numpy", "torch", "transformers", "sentence-transformers", "spacy", "groq",
                 "openai", "rank-bm25", "PyYAML", "faiss-cpu", "scipy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    hardware = {"machine": platform.machine(), "processor": platform.processor(),
                "platform": platform.platform(), "cpu_count": os.cpu_count(), "gpu": []}
    if versions["torch"] is None:
        hardware["gpu_status"] = "torch unavailable; GPU details not queried"
    else:
        try:
            import torch
            hardware["gpu"] = [{"name": torch.cuda.get_device_properties(i).name,
                                "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                               for i in range(torch.cuda.device_count())]
            hardware["cuda_runtime"] = torch.version.cuda
        except (ImportError, RuntimeError, OSError) as exc:
            hardware["gpu_status"] = type(exc).__name__
    return {"python": platform.python_version(), "packages": versions, "hardware": hardware}


def resolve_options(config_path: Path, config: dict, options: dict | None) -> dict:
    given = options or {}
    if given.keys() - DEFAULTS.keys():
        raise ValueError(f"Unknown runtime options: {sorted(given.keys() - DEFAULTS.keys())}")
    result = {**DEFAULTS, **given}
    if result["provider"] not in ("groq", "openai", "local"):
        raise ValueError("Unsupported provider")
    for name in ("model", "model_revision"):
        if not isinstance(result[name], str) or not result[name].strip():
            raise ValueError(f"{name} must be nonempty")
    if result["verifier"] not in ("llm", "nli", "ensemble"):
        raise ValueError("This runner supports prompted/NLI verifiers; trained variants are a later integration")
    for key in ("top_k", "n_samples", "max_repair_iterations"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if result["repair_guard"] not in ("legacy", "grounded_v1"):
        raise ValueError("repair_guard must be legacy or grounded_v1")
    if result["reasoning_effort"] not in ("low", "medium", "high"):
        raise ValueError("reasoning_effort must be low, medium or high")
    if type(result["gptoss_min_completion_tokens"]) is not int or not 1024 <= result["gptoss_min_completion_tokens"] <= 65536:
        raise ValueError("gptoss_min_completion_tokens must be between 1024 and 65536")
    if result["repair_policy"] not in ("rewrite+hedge", "rewrite+remove", "rewrite-only"):
        raise ValueError("Unsupported repair policy")
    if (type(result["consistency_threshold"]) not in (int, float)
            or not 0 <= result["consistency_threshold"] <= 1):
        raise ValueError("consistency_threshold must be in [0,1]")
    from src.retrieve import validate_retrieval_options
    validate_retrieval_options(result["query_strategy"], result["retrieval_max_queries"],
                               result["retrieval_max_passages"], result["top_k"])
    if type(result["tracking_max_claims"]) is not int or not 1 <= result["tracking_max_claims"] <= 512:
        raise ValueError("tracking_max_claims must be between 1 and 512")
    result["index_path"] = str((config_path.parent / config["corpus"]["index_path"]).resolve())
    for key in ("faiss_index_path", "passages_path"):
        if result[key] is not None:
            result[key] = str(Path(result[key]).resolve())
    return result


def preflight(systems: list[dict], options: dict) -> dict:
    from urllib.parse import urlsplit
    endpoint = endpoint_identity(options["provider"])
    if options["provider"] != "local":
        url = urlsplit(endpoint)
        if (url.scheme not in ("http", "https") or not url.netloc or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("Endpoint must be an HTTP(S) URL without credentials/query/fragment")
    if any(system["id"] != "b0" for system in systems):
        credential = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}.get(options["provider"])
        if credential and not os.environ.get(credential):
            raise ValueError(f"Required credential is absent: {credential}")
        module = {"groq": "groq", "openai": "openai", "local": "transformers"}[options["provider"]]
        import importlib.util
        if importlib.util.find_spec(module) is None:
            raise ValueError(f"Install provider dependency: {module}")
    paths = {}
    if options["provider"] != "local" and options["model_revision"] != "hosted-unpinned":
        raise ValueError("Hosted revisions cannot be passed separately; select a versioned model ID")
    if options["provider"] == "local":
        model_path = Path(options["model"])
        if model_path.is_dir():
            for file in model_path.rglob("*"):
                if file.is_file():
                    paths["local_model/" + str(file.relative_to(model_path))] = file.resolve()
            if not any(key.startswith("local_model/") for key in paths):
                raise ValueError("Local model directory is empty")
        elif (len(options["model_revision"]) != 40
              or any(char not in "0123456789abcdef" for char in options["model_revision"].lower())):
            raise ValueError("Remote local-model loading requires a 40-character model commit revision")
    if any(system["retrieval"] != "none" for system in systems):
        paths["index_path"] = Path(options["index_path"])
    if any(system["retrieval"].startswith(("dense", "hybrid")) for system in systems):
        for key in ("faiss_index_path", "passages_path"):
            if not options[key]:
                raise ValueError("Dense/hybrid retrieval requires FAISS and passages paths")
            paths[key] = Path(options[key])
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Required retrieval artifact missing: {path}")
    return {key: {"path": str(path), "sha256": file_hash(path)} for key, path in paths.items()}


@contextmanager
def pinned_segmenter(model):
    import src.decompose as decomposition
    previous = decomposition._nlp
    decomposition._nlp = model
    try:
        yield
    finally:
        decomposition._nlp = previous


@contextmanager
def run_lock(directory: Path):
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process is using this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _read_attempts(directory, sources, systems, task, inventories):
    successes, counters = {}, {}
    for path in sorted((directory / "attempts").glob("*/*.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        result = wrapper["result"]
        if wrapper.get("sha256") != fingerprint(result):
            raise ValueError(f"Attempt checksum mismatch: {path}")
        system, record_id = result["system"], result["input"]["id"]
        if system not in systems or record_id not in sources:
            raise ValueError("Attempt has unknown system or record ID")
        key = (system, record_id)
        expected = fingerprint([system, record_id])
        attempt = result.get("attempt")
        if (type(attempt) is not int or attempt < 1 or path.parent.name != expected
                or path.name != f"{attempt:06d}.json"):
            raise ValueError("Attempt path/number mismatch")
        validate_result(result, sources[record_id], system, task, inventories.get(record_id))
        if attempt != counters.get(key, 0) + 1:
            raise ValueError("Missing or out-of-order attempt")
        trace = (directory / result["trace_path"]).resolve()
        if not trace.is_relative_to(directory / "traces") or not trace.is_file():
            raise ValueError("Missing or invalid trace path")
        if result["trace_sha256"] != file_hash(trace):
            raise ValueError("Trace checksum mismatch")
        if key in successes:
            raise ValueError("An attempt follows an already successful output")
        counters[key] = max(attempt, counters.get(key, 0))
        if result["status"] == "success":
            successes[key] = result
    return successes, counters


def _export(directory, records, systems, successes):
    # Per-record attempts are authoritative. Aggregates are replaceable views.
    for system in systems:
        path = directory / "outputs" / f"{system}.jsonl"
        data = b"".join(canonical(successes[(system, record.id)]) + b"\n" for record in records
                        if (system, record.id) in successes)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    errors = []
    for path in sorted((directory / "attempts").glob("*/*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))["result"]
        if result["status"] == "error":
            errors.append(result)
    path = directory / "errors.jsonl"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(b"".join(canonical(item) + b"\n" for item in errors))
    os.replace(temporary, path)


def run_benchmark(*, config_path: Path, inputs_path: Path, splits_path: Path,
                  output_dir: Path, task: str, split: str, inventory_path: Path | None = None,
                  options: dict | None = None, resume: bool = False) -> dict:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = load_config(config_path)
    if task not in ("repair", "claim_verification", "answer_detection") or split not in ("dev", "test"):
        raise ValueError("Invalid task or split")
    if (task == "claim_verification") != (inventory_path is not None):
        raise ValueError("Supply --inventory exactly for claim_verification")
    systems = [system for system in config["systems"] if task in system["tasks"]]
    if not systems:
        raise ValueError("No configured systems support the requested task")
    records = load_inputs(inputs_path, split)
    splits, note = read_splits(splits_path)
    if splits is None or {record.id for record in records} != splits[split]:
        raise ValueError(f"Input IDs must exactly match the selected split: {note}")
    if any(len(splits[name]) != config["splits"][name] for name in ("dev", "test")):
        raise ValueError("Split counts differ from declared protocol")
    resolved = resolve_options(config_path, config, options)
    assets = preflight(systems, resolved)
    segmenter = require_segmenter(config)
    inventories = read_inventory(inventory_path, records, segmenter) if inventory_path else {}
    from src.experiments.provenance import freeze_environment, save_reproduction, validate_reproduction
    identity = {"schema_version": 1,
                "reproduction_environment_sha256": fingerprint(freeze_environment()), "config": config, "config_sha256": file_hash(config_path),
                "options": resolved, "task": task, "split": split,
                "endpoint": endpoint_identity(resolved["provider"]),
                "input_sha256": file_hash(inputs_path), "splits_sha256": file_hash(splits_path),
                "inventory_sha256": file_hash(inventory_path) if inventory_path else None,
                "code": code_identity(), "environment": environment(), "assets": assets,
                "input_ids": [record.id for record in records],
                "auxiliary_models": {"nli": "cross-encoder/nli-deberta-v3-small",
                                     "dense_encoder": "BAAI/bge-small-en-v1.5",
                                     "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                                     "revision_policy": "upstream defaults; unpinned"}}
    run_id = fingerprint(identity)
    if resume and not output_dir.is_dir():
        raise ValueError("Cannot resume a missing run directory")
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=False)
    with run_lock(output_dir):
        manifest_path = output_dir / "manifest.json"
        if resume:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("run_id") != run_id or manifest.get("identity") != identity:
                raise ValueError("Resume identity changed (inputs, code, model, settings, or environment)")
            validate_reproduction(output_dir, manifest)
        else:
            for folder in ("attempts", "outputs", "traces"):
                (output_dir / folder).mkdir()
            reproduction = save_reproduction(ROOT, output_dir, identity)
            manifest = {"schema_version": 1, "run_id": run_id, "identity": identity,
                        "reproduction_files_sha256": reproduction,
                        "created_utc": now(), "benchmark_complete": False,
                        "cache_policy": "run-local SQLite; B2 sampling and retried failed records bypass LLM cache",
                        "token_usage": "completion_metadata trace events; cached responses do not incur new usage",
                        "dirty_worktree": bool(subprocess.run(["git", "status", "--porcelain"],
                            cwd=ROOT, capture_output=True, text=True).stdout.strip()),
                        "note": "Execution only; independent evaluation and metrics are separate."}
        sources = {record.id: record for record in records}
        system_map = {system["id"]: system for system in systems}
        successes, counters = _read_attempts(output_dir, sources, system_map, task, inventories)
        if manifest.get("successful_outputs", 0) > len(successes):
            raise ValueError("Previously successful output artifacts are missing")
        manifest.update(status="running", updated_utc=now())
        atomic_json(manifest_path, manifest)
        # Caches are private to this run, and import-time environment overrides are restored.
        from src import llm, retrieve
        old_env = {name: os.environ.get(name) for name in ("LLM_CACHE_PATH", "RETRIEVAL_CACHE_PATH")}
        llm.reset_connection()
        previous_stores = (retrieve._bm25_store, retrieve._dense_store, retrieve._cross_encoder)
        retrieve._bm25_store, retrieve._dense_store, retrieve._cross_encoder = {}, {}, None
        previous_retrieval_conn = retrieve._retrieval_conn
        retrieve._retrieval_conn = None
        for name, filename in (("LLM_CACHE_PATH", "llm_cache.db"),
                               ("RETRIEVAL_CACHE_PATH", "retrieval_cache.db")):
            os.environ[name] = str(output_dir / filename)
        try:
            with pinned_segmenter(segmenter):
                for system in systems:
                    for source in records:
                        key = (system["id"], source.id)
                        if key in successes:
                            continue
                        attempt = counters.get(key, 0) + 1
                        trace_path = output_dir / "traces" / f"{uuid.uuid4().hex}.jsonl"
                        started = time.monotonic()
                        base_result = {"schema_version": 1, "system": system["id"], "task": task,
                                  "input": {field: getattr(source, field) for field in IDENTITY},
                                  "attempt": attempt, "started_utc": now()}
                        result = dict(base_result)
                        with trace_path.open("xb") as trace:
                            def record_call(call):
                                trace.write(canonical(call) + b"\n")
                                trace.flush()
                                os.fsync(trace.fileno())
                            fresh = copy.deepcopy(source)
                            fresh.claims = copy.deepcopy(inventories.get(source.id, []))
                            seed = (config["seed"] + int(fingerprint([system["id"], source.id])[:8], 16)) % (2**31)
                            random.seed(seed)
                            import numpy as np
                            np.random.seed(seed)
                            try:
                                import torch
                                torch.manual_seed(seed)
                                if torch.cuda.is_available():
                                    torch.cuda.manual_seed_all(seed)
                            except ImportError:
                                pass
                            try:
                                with execution(record=record_call, seed=seed, refresh=attempt > 1,
                                               revision=resolved["model_revision"] if resolved["provider"] == "local" else None,
                                               settings=resolved):
                                    payload = adapters.execute(system, fresh, task, resolved)
                                if not isinstance(payload, dict) or set(payload) != {
                                        "record", "native", "retrieved_evidence"}:
                                    raise ValueError("Invalid adapter payload fields")
                                result.update(payload, status="success", error=None,
                                              elapsed_seconds=time.monotonic() - started)
                                validate_result(result, source, system["id"], task, inventories.get(source.id))
                            except Exception as exc:
                                result = {**base_result,
                                          "status": "error", "error": {"type": type(exc).__name__,
                                          "message": str(exc)}, "elapsed_seconds": time.monotonic() - started}
                        result.update(trace_path=str(trace_path.relative_to(output_dir)),
                                      trace_sha256=file_hash(trace_path), seed=seed)
                        attempt_dir = output_dir / "attempts" / fingerprint(list(key))
                        attempt_dir.mkdir(exist_ok=True)
                        atomic_json(attempt_dir / f"{attempt:06d}.json",
                                    {"result": result, "sha256": fingerprint(result)})
                        counters[key] = attempt
                        if result["status"] == "success":
                            successes[key] = result
        finally:
            llm.reset_connection()
            if retrieve._retrieval_conn is not None:
                retrieve._retrieval_conn.close()
            retrieve._retrieval_conn = previous_retrieval_conn
            retrieve._bm25_store, retrieve._dense_store, retrieve._cross_encoder = previous_stores
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            _export(output_dir, records, system_map, successes)
            expected = len(records) * len(systems)
            manifest.update(status="execution_complete" if len(successes) == expected else "incomplete",
                            expected_outputs=expected, successful_outputs=len(successes),
                            updated_utc=now())
            atomic_json(manifest_path, manifest)
        return manifest
