#!/bin/bash
# Script to compile the architecture diagram

cd "$(dirname "$0")"

echo "Compiling FAA-Net architecture diagram..."
pdflatex -interaction=nonstopmode faanet_architecture.tex

# Clean up auxiliary files
rm -f faanet_architecture.aux faanet_architecture.log

echo "Done! Generated: faanet_architecture.pdf"
echo ""
echo "To use in your paper, add this to access.tex:"
echo ""
echo "\\begin{figure*}[!t]"
echo "\\centering"
echo "\\includegraphics[width=0.95\\textwidth]{faanet_architecture.pdf}"
echo "\\caption{FAA-Net architecture overview...}"
echo "\\label{fig:architecture}"
echo "\\end{figure*}"
