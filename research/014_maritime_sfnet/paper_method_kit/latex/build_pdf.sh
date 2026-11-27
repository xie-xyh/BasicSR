#!/usr/bin/env bash
set -euo pipefail

export PATH="/Library/TeX/texbin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "[ERROR] pdflatex not found in /Library/TeX/texbin or PATH."
  echo "        If 'brew install --cask mactex' is still waiting for sudo password, finish it first."
  exit 1
fi

echo "[1/3] Compiling maritime_sfnet_method.tex (pass 1)..."
pdflatex -interaction=nonstopmode -halt-on-error maritime_sfnet_method.tex > build_pass1.log
echo "[2/3] Compiling maritime_sfnet_method.tex (pass 2, resolving cross-refs)..."
pdflatex -interaction=nonstopmode -halt-on-error maritime_sfnet_method.tex > build_pass2.log

echo "[3/3] Checking LaTeX warnings..."
grep -E "Warning|Undefined|Overfull \\\\hbox" maritime_sfnet_method.log || echo "No critical warnings found."

if command -v pdftoppm >/dev/null 2>&1; then
  mkdir -p page_previews
  pdftoppm -png -r 150 maritime_sfnet_method.pdf page_previews/page
  echo "Rendered PNG page previews to $SCRIPT_DIR/page_previews/"
fi

echo "Done! PDF generated at: $SCRIPT_DIR/maritime_sfnet_method.pdf"
