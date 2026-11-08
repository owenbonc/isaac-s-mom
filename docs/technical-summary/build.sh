#!/bin/bash
# Builds isaacs-mom-technical-summary.pdf: charts from data.json, the system diagram, then the one-page paper.
set -euo pipefail
cd "$(dirname "$0")"
python3 make_figures.py 2>/dev/null
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build fig_system.tex >build/fig_system.latexmk.txt 2>&1 || { tail -30 build/fig_system.log; exit 1; }
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build summary.tex >build/summary.latexmk.txt 2>&1 || { tail -30 build/summary.log; exit 1; }
pages=$(sed -n 's/.*Output written on .* (\([0-9]*\) page.*/\1/p' build/summary.log | tail -1)
cp build/summary.pdf isaacs-mom-technical-summary.pdf
echo "isaacs-mom-technical-summary.pdf: ${pages} page(s)"
[ "$pages" = "1" ] || { echo "WARNING: more than one page"; exit 2; }
