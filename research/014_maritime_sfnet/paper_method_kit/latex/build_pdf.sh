#!/usr/bin/env bash
set -euo pipefail

export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "[ERROR] pdflatex not found in /Library/TeX/texbin or PATH."
  exit 1
fi

echo "[1/3] Compiling maritime_sfnet_method.tex (pass 1)..."
pdflatex -interaction=nonstopmode -halt-on-error maritime_sfnet_method.tex > build_pass1.log
echo "[2/3] Compiling maritime_sfnet_method.tex (pass 2, resolving cross-refs)..."
pdflatex -interaction=nonstopmode -halt-on-error maritime_sfnet_method.tex > build_pass2.log

echo "[3/3] Checking LaTeX warnings..."
grep -E "Warning|Undefined|Overfull \\\\hbox" maritime_sfnet_method.log || echo "No critical warnings found."

mkdir -p page_previews
rm -f page_previews/page*.png
if command -v pdftoppm >/dev/null 2>&1; then
  pdftoppm -png -r 150 maritime_sfnet_method.pdf page_previews/page
elif command -v gs >/dev/null 2>&1; then
  gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r150 \
     -sOutputFile=page_previews/page-%02d.png maritime_sfnet_method.pdf
fi
ls -la page_previews/
echo "Done! PDF generated at: $SCRIPT_DIR/maritime_sfnet_method.pdf"
