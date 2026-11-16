# Project report

- `project_report.tex`: standalone LaTeX source; upload this file to Overleaf.
- `project_report.pdf`: compiled report.

The title page includes the four supplied names and roll numbers. Institution and supervisor macros are intentionally empty; fill them in at the top of the source if required.

Compile twice from the repository root:

```bash
mkdir -p /tmp/hallucination-report-build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/hallucination-report-build docs/report/project_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/hallucination-report-build docs/report/project_report.tex
cp /tmp/hallucination-report-build/project_report.pdf docs/report/project_report.pdf
```

The source uses standard LaTeX packages and does not require external figures or a bibliography file. Hyperlinks are enabled when the required hyperref dependency is available; a minimal installation still compiles without them.

## Results to update before submission

The report uses the five-answer v4b pilot described in `docs/project_handoff.md`. The underlying private bundles were absent from this checkout, so those historical findings were not recomputed here. Older aggregate performance tables are not copied into this report.

When the team provides a new completed run, update the results table, experimental configuration, abstract, and conclusion together. Include the exact split, answer/claim counts, model, extraction mode, retrieval method, independent-review procedure, and failure coverage. Keep self-verification diagnostics separate from independently assessed factual correction. Remove the pending-results table if no additional experiment will be included.

No existing reports, Colab files, results, or experiment artifacts were modified to create this report.

The architecture section explains component inputs, outputs, algorithms, implementation files, guard behavior, claim tracking, and stopping conditions. A dedicated baseline section describes B0–B7 and their native capabilities; it distinguishes repository adaptations from exact published reproductions and does not add unmeasured baseline scores.
