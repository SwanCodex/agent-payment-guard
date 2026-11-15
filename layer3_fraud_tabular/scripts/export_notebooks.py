"""Turns downloaded Kaggle notebooks (.ipynb) into plain .py files (percent format) in notebooks/.
Only the code and markdown are kept (no outputs), so no notebook JSON goes into git.

usage:  python scripts/export_notebooks.py <folder that holds the downloaded .ipynb files>
"""
import json
import pathlib
import sys


def convert(nb_path, out_dir):
    nb = json.loads(pathlib.Path(nb_path).read_text(encoding="utf-8"))
    parts = []
    for cell in nb["cells"]:
        src = "".join(cell["source"]).rstrip()
        if not src:
            continue
        if cell["cell_type"] == "markdown":
            parts.append("# %% [markdown]\n" + "\n".join(("# " + line) if line else "#" for line in src.split("\n")))
        elif cell["cell_type"] == "code":
            parts.append("# %%\n" + src)
    out = pathlib.Path(out_dir) / (pathlib.Path(nb_path).stem.replace("-", "_") + ".py")
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    folder = pathlib.Path(sys.argv[1])
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for nb_file in sorted(folder.glob("*.ipynb")):
        print("written:", convert(nb_file, out_dir))
