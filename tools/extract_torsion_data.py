"""Extracts the ETKDG torsion-preference SMARTS tables from the RDKit sources.

Usage: py -I tools/extract_torsion_data.py <rdkit-source-root>

Reads Code/GraphMol/ForceFieldHelpers/CrystalFF/torsionData/*.in (C++ string
literal concatenations) and writes the concatenated table text, one pattern
per line exactly as RDKit's parser sees it, to src/distgeom/data/*.txt.

The data is part of the RDKit (BSD-3-Clause, Copyright (C) 2017-2026 Sereina
Riniker and other RDKit contributors).
"""
import os
import sys

FILES = {
    "torsionPreferences_v1.in": "torsion_prefs_v1.txt",
    "torsionPreferences_v2.in": "torsion_prefs_v2.txt",
    "torsionPreferences_smallrings.in": "torsion_prefs_smallrings.txt",
    "torsionPreferences_macrocycles.in": "torsion_prefs_macrocycles.txt",
}


def string_literals(src):
    """Concatenation of all C++ string literals after the first '=' (comments skipped)."""
    i = src.index("=")
    out = []
    n = len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            i = src.index("\n", i)
        elif src.startswith("/*", i):
            i = src.index("*/", i) + 2
        elif c == '"':
            i += 1
            buf = []
            while src[i] != '"':
                if src[i] == "\\":
                    e = src[i + 1]
                    buf.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"'}[e])
                    i += 2
                else:
                    buf.append(src[i])
                    i += 1
            out.append("".join(buf))
            i += 1
        elif c == ";":
            break
        else:
            i += 1
    return "".join(out)


def main():
    root = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(here, "..", "src", "distgeom", "data")
    os.makedirs(dst, exist_ok=True)
    src_dir = os.path.join(root, "Code", "GraphMol", "ForceFieldHelpers", "CrystalFF", "torsionData")
    for f, o in FILES.items():
        with open(os.path.join(src_dir, f), encoding="utf-8") as fh:
            text = string_literals(fh.read())
        with open(os.path.join(dst, o), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f, "->", o, text.count("\n"), "lines")


if __name__ == "__main__":
    main()
