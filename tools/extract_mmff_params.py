#!/usr/bin/env python
"""Extract the MMFF94 parameter tables embedded in RDKit's
Code/ForceField/MMFF/Params.cpp into src/ff/mmff/data/*.par.

Usage:
    py tools/extract_mmff_params.py <path to Params.cpp> [out dir]

The tables are the C++ string literals ``defaultMMFF*`` (and the
``defaultMMFFAngleData[]`` array, terminated by "EOS"); they are written
verbatim (tab separated, '*' comment lines) so the Rust parser can follow
RDKit's own parsing logic line by line.
"""
import os
import re
import sys

TABLES = {
    "defaultMMFFDef": "def.par",
    "defaultMMFFProp": "prop.par",
    "defaultMMFFPBCI": "pbci.par",
    "defaultMMFFChg": "chg.par",
    "defaultMMFFBond": "bond.par",
    "defaultMMFFBndk": "bndk.par",
    "defaultMMFFHerschbachLaurie": "herschbach_laurie.par",
    "defaultMMFFCovRadPauEle": "covrad_pauele.par",
    "defaultMMFFAngleData[]": "angle.par",
    "defaultMMFFStbn": "stbn.par",
    "defaultMMFFDfsb": "dfsb.par",
    "defaultMMFFOop": "oop.par",
    "defaultMMFFsOop": "oop_mmffs.par",
    "defaultMMFFTor": "tor.par",
    "defaultMMFFsTor": "tor_mmffs.par",
    "defaultMMFFVdW": "vdw.par",
}

LIT = re.compile(r'"((?:[^"\\]|\\.)*)"')


def unescape(s):
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            n = s[i + 1]
            out.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'"}[n])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def main():
    src = open(sys.argv[1], encoding="utf-8").read()
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "src", "ff", "mmff", "data")
    os.makedirs(out_dir, exist_ok=True)
    for var, fname in TABLES.items():
        decl = "const std::string %s =" % var
        start = src.index(decl) + len(decl)
        end = src.index(";\n", start)
        body = src[start:end]
        parts = [unescape(m) for m in LIT.findall(body)]
        if var.endswith("[]"):
            parts = parts[: parts.index("EOS")]
        text = "".join(parts)
        assert text.endswith("\n"), var
        with open(os.path.join(out_dir, fname), "w", newline="\n", encoding="utf-8") as fh:
            fh.write(text)
        print("%-28s -> %-22s %6d lines" % (var, fname, text.count("\n")))


if __name__ == "__main__":
    main()
