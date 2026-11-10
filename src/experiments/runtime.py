"""Context-local strict parsing and request recording for benchmark execution."""
from contextlib import contextmanager
from contextvars import ContextVar

strict = ContextVar("benchmark_strict", default=False)
recorder = ContextVar("benchmark_recorder", default=None)
run_seed = ContextVar("benchmark_seed", default=None)
bypass_cache = ContextVar("benchmark_bypass_cache", default=False)
request_settings = ContextVar("benchmark_request_settings", default=None)
model_revision = ContextVar("benchmark_model_revision", default=None)


@contextmanager
def execution(*, record=None, seed=42, refresh=False, revision=None, settings=None):
    tokens = [(strict, strict.set(True)), (recorder, recorder.set(record)),
              (run_seed, run_seed.set(seed)), (bypass_cache, bypass_cache.set(refresh)),
              (model_revision, model_revision.set(revision)),
              (request_settings, request_settings.set(settings))]
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)
