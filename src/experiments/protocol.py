"""Versioned evaluation configuration and native system capabilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Capability:
    tasks: tuple[str, ...]
    retrieval: tuple[str, ...]
    native_scores: tuple[str, ...] = ()
    prerequisite: str | None = None


# These describe the repository implementations, not the original papers.
SYSTEMS = {
    "pipeline": Capability(("claim_verification", "repair"),
                           ("bm25", "fts5", "dense", "hybrid", "hybrid+rerank"),
                           ("self_verification_scr",)),
    "b0": Capability(("repair",), ("none",)),
    "b1": Capability(("answer_detection",), ("none",), ("answer_confidence",)),
    "b2": Capability(("repair",), ("none",), ("sentence_consistency",)),
    "b3": Capability(("repair",), ("none",)),
    "b4": Capability(("repair",), ("none",)),
    "b5": Capability(("repair",), ("bm25",)),
    "b6": Capability(("repair",), ("bm25", "fts5", "dense", "hybrid", "hybrid+rerank"),
                     prerequisite="pipeline_detector"),
    "b7": Capability(("claim_verification", "repair"), ("bm25",)),
}
TASKS = {"answer_detection", "claim_verification", "repair"}


def _keys(value: Any, required: set[str], optional: set[str], where: str) -> dict:
    if not isinstance(value, dict) or required - value.keys() or value.keys() - required - optional:
        raise ValueError(f"{where}: required keys {sorted(required)}; optional {sorted(optional)}")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected nonempty text")
    return value


def _count(value: Any, where: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where}: expected integer >= {minimum}")
    return value


def validate_config(config: Any, *, require_sources: bool = False) -> dict:
    """Reject typos, unsupported system/task combinations, and ambiguous quotas."""
    cfg = _keys(config, {"schema_version", "seed", "splits", "corpus", "segmentation",
                         "systems", "sources"}, set(), "configuration")
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("Unsupported schema_version (expected 1)")
    _count(cfg["seed"], "seed")
    splits = _keys(cfg["splits"], {"dev", "test"}, set(), "splits")
    for split in splits:
        _count(splits[split], f"splits.{split}", 1)
    corpus = _keys(cfg["corpus"], {"snapshot", "index_path"}, set(), "corpus")
    _text(corpus["snapshot"], "corpus.snapshot")
    _text(corpus["index_path"], "corpus.index_path")
    segmentation = _keys(cfg["segmentation"], {"backend", "model", "version"}, set(), "segmentation")
    if segmentation["backend"] != "spacy":
        raise ValueError("segmentation.backend must be spacy (no implicit fallback)")
    _text(segmentation["model"], "segmentation.model")
    _text(segmentation["version"], "segmentation.version")
    if not isinstance(cfg["systems"], list) or not cfg["systems"]:
        raise ValueError("systems must be a nonempty list")
    seen = set()
    for system in cfg["systems"]:
        item = _keys(system, {"id", "tasks", "retrieval"}, {"detector"}, "system")
        name = _text(item["id"], "system.id")
        if name not in SYSTEMS or name in seen:
            raise ValueError(f"Unknown or duplicate system: {name}")
        seen.add(name)
        capability = SYSTEMS[name]
        tasks = item["tasks"]
        if (not isinstance(tasks, list) or not tasks
                or any(not isinstance(task, str) for task in tasks)
                or len(tasks) != len(set(tasks)) or not set(tasks) <= set(capability.tasks)):
            raise ValueError(f"{name}: unsupported or duplicate tasks")
        if item["retrieval"] not in capability.retrieval:
            raise ValueError(f"{name}: unsupported retrieval method")
        if capability.prerequisite:
            if item.get("detector") != capability.prerequisite:
                raise ValueError(f"{name}: detector must be {capability.prerequisite}")
        elif "detector" in item:
            raise ValueError(f"{name}: detector is not applicable")
    if not isinstance(cfg["sources"], list) or (require_sources and not cfg["sources"]):
        raise ValueError("sources must be a list; annotation preparation requires at least one source")
    source_ids = set()
    totals = {"dev": 0, "test": 0}
    for source in cfg["sources"]:
        item = _keys(source, {"id", "path", "version", "quotas"}, set(), "source")
        name = _text(item["id"], "source.id")
        if name in source_ids:
            raise ValueError(f"Duplicate source: {name}")
        source_ids.add(name)
        _text(item["path"], f"{name}.path")
        _text(item["version"], f"{name}.version")
        quotas = _keys(item["quotas"], {"dev", "test"}, set(), f"{name}.quotas")
        for split in totals:
            totals[split] += _count(quotas[split], f"{name}.quotas.{split}")
    if cfg["sources"] and totals != splits:
        raise ValueError(f"Source quotas {totals} must exactly match split sizes {splits}")
    return cfg


def load_config(path: Path, *, require_sources: bool = False) -> dict:
    import yaml
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")),
                           require_sources=require_sources)


def require_segmenter(config: dict):
    """Load the declared model only at stages that need sentence boundaries."""
    from importlib.metadata import version
    spec = config["segmentation"]
    try:
        import spacy
        installed = version(spec["model"].replace("_", "-"))
        if installed != spec["version"]:
            raise ValueError(f"Segmenter version {installed} != declared {spec['version']}")
        model = spacy.load(spec["model"])
        if not any(name in model.pipe_names for name in ("parser", "senter", "sentencizer")):
            raise ValueError("Declared model has no sentence boundary component")
        return model
    except (ImportError, OSError) as exc:
        raise ValueError(f"Install declared segmenter {spec['model']}=={spec['version']}; no fallback") from exc
