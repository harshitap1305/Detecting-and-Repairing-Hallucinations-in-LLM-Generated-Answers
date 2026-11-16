# Project report

- `project_report.tex`: LaTeX report source. For Overleaf, upload it together with the four retained PNG files from `results/figures/`, placing the PNGs in a `figures/` folder.
- `project_report.pdf`: compiled report.

The title page includes the four supplied names and roll numbers. Institution and supervisor macros are intentionally empty; fill them in at the top of the source if required.

Compile twice from the repository root:

```bash
mkdir -p /tmp/hallucination-report-build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/hallucination-report-build docs/report/project_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/hallucination-report-build docs/report/project_report.tex
cp /tmp/hallucination-report-build/project_report.pdf docs/report/project_report.pdf
```

The source uses standard LaTeX packages, including `graphicx`, and embeds four PNG figures. Its bibliography remains in the LaTeX source. Figure lookup supports compilation from the repository root, from `docs/report/`, or on Overleaf with a `figures/` folder. Hyperlinks are enabled when `hyperref.sty` is available; a minimal installation still compiles without them.

## Results to update before submission

The report uses the five-answer v4b pilot described in `docs/project_handoff.md`. Its private gold and final bundle are present locally and passed offline recomputation on 16 November 2026:

```bash
python3 scripts/audit_baseline.py \
  --bundle results/evaluation/colab_followup_v4b/pilot_final_v1 \
  --gold data/gold/pilot_dev.jsonl
```

The audit returned `valid_final_task_bundle`, `recomputed: true`, five development records, and systems `b0` and `pipeline`. This validates the saved bundle and metrics; it does not repeat inference or independently verify the annotations. Private artifacts are ignored by Git and require separate transfer for a fresh clone.

The report now includes the recorded pilot configuration, B0 comparison, review agreement, metric denominators, descriptive bootstrap intervals, and a development roadmap informed by saved diagnostics. The historical run used entity planner version 2; current code/options use version 3. Use the embedded manifest when describing the pilot.

The report includes original chart assets 1, 3, 4, and 6 in a dedicated team-supplied experimental-results section. The user identified their source as a teammate's Drive results. Captions and the source note preserve this attribution; local offline recomputation applies to the separate v4b pilot. The [figure provenance review](figure_provenance.md) records the source history and the raw artifacts needed for additional recomputation.

Original figures 2 (synthetic threshold heatmap) and 5 (illustrative training curves) were removed from `results/figures/` and their generation functions/calls were removed from `scripts/plot_results.py`. Retained asset filenames preserve the original numbers; LaTeX assigns consecutive figure numbers, including the architecture diagram.

When the team provides a new completed run, update the results table, experimental configuration, abstract, and conclusion together. Include the exact split, answer/claim counts, model, extraction mode, retrieval method, independent-review procedure, and failure coverage. Keep self-verification diagnostics separate from independently assessed factual correction. Keep the team-supplied aggregate experiment descriptions and locally recomputed pilot clearly attributed when adding new artifacts.

The figure update changes the report source/PDF, plotting script, figure assets, and accompanying documentation. Pipeline and training code are unchanged.

The architecture section explains component inputs, outputs, algorithms, implementation files, guard behavior, claim tracking, and stopping conditions. A dedicated baseline section describes B0–B7 and their native capabilities; it distinguishes repository adaptations from exact published reproductions and does not add unmeasured baseline scores.

The report uses a contribution-focused presentation with design insights, a development roadmap, and evidence-transparency details. Measured outcomes and experiment scope remain explicit.
