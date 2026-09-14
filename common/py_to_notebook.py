"""
Utility: convert an annotated .py training script into a fully-executed .ipynb with
visible outputs. Splits the script into cells at '# ---- ... ----' section-divider
comments (already used throughout this repo's scripts), then runs nbconvert --execute
so the notebook ships with real captured outputs (prints, plots) baked in.

Usage:
    python common/py_to_notebook.py <script.py> <output.ipynb> [--no-execute]
"""
import re
import subprocess
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

DIVIDER_RE = re.compile(r"^# -{10,}\s*$")


def split_into_cells(source: str):
    lines = source.splitlines()
    cells = []
    current = []
    pending_title = None

    def flush():
        nonlocal current, pending_title
        text = "\n".join(current).strip("\n")
        if text.strip():
            if pending_title:
                cells.append(new_markdown_cell(f"### {pending_title}"))
            cells.append(new_code_cell(text))
        current = []
        pending_title = None

    i = 0
    while i < len(lines):
        line = lines[i]
        if DIVIDER_RE.match(line):
            # pattern: divider / title comment / divider -> section header
            if i + 2 < len(lines) and DIVIDER_RE.match(lines[i + 2]) and lines[i + 1].startswith("#"):
                flush()
                pending_title = lines[i + 1].lstrip("#").strip()
                i += 3
                continue
        current.append(line)
        i += 1
    flush()
    return cells


def main():
    src_path = sys.argv[1]
    out_path = sys.argv[2]
    execute = "--no-execute" not in sys.argv

    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Strip the module docstring into a markdown title cell
    docstring_match = re.match(r'^"""(.*?)"""\s*\n', source, re.DOTALL)
    nb_cells = []
    if docstring_match:
        nb_cells.append(new_markdown_cell(docstring_match.group(1).strip()))
        source = source[docstring_match.end():]

    nb_cells.extend(split_into_cells(source))
    nb = new_notebook(cells=nb_cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Wrote {out_path} with {len(nb_cells)} cells")

    if execute:
        subprocess.run([
            sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
            "--inplace", "--ExecutePreprocessor.timeout=5400", out_path,
        ], check=True)
        print(f"Executed {out_path}")


if __name__ == "__main__":
    main()
