#!/usr/bin/env python3
"""Convert SVG plots under results/ to PNG format using PyMuPDF.
Handles encoding conversion from CP1252 to UTF-8 if necessary.
"""
import os
import sys

try:
    import fitz
except ImportError:
    print("Error: PyMuPDF (fitz) is not installed. Please run this script using 'uv':")
    print("  uv run --with pymupdf python scripts/convert_svgs.py")
    print("Or install it manually via pip:")
    print("  pip install pymupdf")
    sys.exit(1)

def convert_svg_to_png(svg_path):
    png_path = os.path.splitext(svg_path)[0] + ".png"
    print(f"Converting: {svg_path} -> {png_path}")
    try:
        # Check encoding and rewrite as UTF-8 if it was CP1252
        try:
            with open(svg_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"  File {svg_path} is not UTF-8. Attempting CP1252...")
            with open(svg_path, "r", encoding="cp1252") as f:
                content = f.read()
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Rewrote {svg_path} as UTF-8.")

        # Render to PNG
        doc = fitz.open(svg_path)
        page = doc.load_page(0)
        # Render at 2x scale for crisp text/lines
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        pix.save(png_path)
        print("  Success")
        return True
    except Exception as e:
        print(f"  Error converting {svg_path}: {e}")
        return False

def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        sys.exit(1)

    svg_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.lower().endswith(".svg"):
                svg_files.append(os.path.join(root, file))

    if not svg_files:
        print("No SVG files found in results/.")
        return

    success_count = 0
    for svg_path in svg_files:
        if convert_svg_to_png(svg_path):
            success_count += 1

    print(f"Converted {success_count}/{len(svg_files)} SVG files to PNG.")

if __name__ == "__main__":
    main()
