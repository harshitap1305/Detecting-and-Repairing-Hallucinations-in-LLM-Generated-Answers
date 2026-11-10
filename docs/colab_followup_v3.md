# Follow-up development pilot v3

Software for follow-up phases 1–5 and the phase 6 run workflow is implemented.
The new live experiment, independent assessment of its outputs and final audited
bundle remain pending. The local workspace has no Groq credential or access to
the user's running Colab session. No new model result is claimed here.

## Run in Colab

1. Open the updated `notebooks/05_run_reviewed_pilot_colab.ipynb`. Preserve the
   earlier `pilot_repair_gptoss_v2` results and final bundle.
2. Use the same active Colab session where possible. The updated setup reuses
   `/content/reviewed_pilot_env`; existing valid gold and the FEVER index are also
   reused. A fresh runtime needs its input ZIP and corpus index restored or built.
3. Push the changes and use that revision, or set `APPLY_CODE_UPDATE = True` in
   the optional update cell and upload the prepared private
   `results/evaluation/pilot_followup_v3_code.zip`. This contains generating source,
   not gold, the FEVER index, credentials or model results. No commit is required
   to use that ZIP. To rebuild it after further source changes:

   ```bash
   python scripts/package_colab_code.py --output results/evaluation/pilot_followup_v3_code.zip
   ```

4. Keep `GROQ_API_KEY` in Colab Secrets. Select the reviewed-input ZIP only if
   prompted by the data cell. Code and data uploads are separate cells.
5. The inspection cell defaults to `MODEL = "openai/gpt-oss-120b"`, low reasoning
   effort, and a minimum 4096-token completion budget. It checks availability;
   there is no silent model substitution. The profile is
   `configs/pilot_followup_v3.options.json`. See [Groq reasoning documentation](https://console.groq.com/docs/reasoning)
   for provider parameter semantics. Only effective supported GPT-OSS options
   are sent to Groq; unrelated models retain their native defaults.
6. Keep `RUN_NAME = "pilot_followup_v3"` and `RESUME = False` for the first run.
   A changed source, model, profile or environment requires a new run name.
   Resume is only for an identical interrupted experiment. The five dev inputs
   and B0/pipeline systems are unchanged; no held-out test execution is included.
7. Run the benchmark and final scoring/review-export cell. The source snapshot
   and installed-package freeze are saved BEFORE inference. The final cell
   computes reference agreement, emits stage diagnostics (with an optional
   comparison to the earlier run), exports a new blinded review packet and makes
   a progress bundle. These diagnostics are not independent factuality metrics.
8. Download the automatically prepared `pilot_followup_v3_results.zip` before
   ending the runtime. It excludes the multi-gigabyte index and caches. An
   intermediate backup is also made after incomplete execution; download that
   manually if the benchmark fails. Local `/content` storage is temporary.

Place the downloaded ZIP in `results/evaluation/` locally. New candidate outputs
need new independent human ratings/adjudication; the earlier ratings cannot bind
to changed outputs. Do not claim improvement until the new outputs have been
reviewed and the new final bundle audited.

## Conservative repair profile

The profile explicitly selects `repair_guard = grounded_v1` and
`repair_policy = rewrite-only`. The standalone repair API retains its named
`legacy` mode for older callers; the pipeline/runner use `grounded_v1` by default.
The existing hedge/remove/rewrite policies remain selectable in runtime options.
There is no hidden switch from minimal factual editing to question completion.

Named-entity and token-overlap guards are preliminary filters. Accepted replacement
relations must additionally pass a structured model evidence/preservation check,
with nonempty valid passage citations. The check can fail; it is labeled a model
guard, not independent human assessment. Malformed checks fail closed. No evidence
means no generation attempt in grounded mode. Exhausted checks retain the input
under the v3 abstention profile. Optional hedging/deletion is unresolved, not a
correction; blanket hedges/deletions cannot alter supported content in mixed
sentences. Original sentence spans preserve untouched whitespace and bytes.
Each prompted sentence has at most two generation/guard attempts by default;
trained generation has one guarded attempt. Diagnostic status and attempt reasons
remain in each pipeline iteration.

## Reproduction and audit artifacts

Each new execution stores `source.zip`, `source_manifest.json`, and
`environment.freeze.json`. The source ZIP captures the actual allowlisted Python
modules, scripts, configuration, requirements, protocol documents and repository
notebook bytes, including dirty edits. Interactive notebook state itself is not
captured; selected runtime options and inputs are frozen separately in execution
identity, and the CLI can reproduce the run without that UI state. Source and
package hashes are checked on resume/load and carried into generated bundles.
No environment variables or credential files are captured. Runtime/platform/API
availability still constrain exact reproduction; hosted weights are unpinned.

GPT-OSS reasoning parameters and effective completion budgets now enter cache
identity and request traces. Successful and empty API responses emit finish
reasons and SDK token usage where available. Cache hits are marked; old billed
usage is not counted as a fresh request. Source snapshots are integrity records,
not proof of annotation correctness or reviewer independence.
