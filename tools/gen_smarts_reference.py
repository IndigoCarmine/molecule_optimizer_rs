"""Generates extra SMARTS reference data (beyond the torsion tables).

Usage:
    py tools/gen_smarts_reference.py

Writes tests/data/smarts/extra_matches.json: for a set of general SMARTS
patterns (rings, ranges, recursion, logic, disconnected queries, patterns
with many matches) the ordered GetSubstructMatches results (uniquify=False
and uniquify=True) on every molecule of tests/data/etkdg/molecules.sdf, plus
the query atom / bond counts and RDKit bond order of every pattern.
Only non-empty match sets are stored.

Requires RDKit (tested with 2026.09.1).
"""
import json
import os

from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SDF = os.path.join(ROOT, "tests", "data", "etkdg", "molecules.sdf")
OUT = os.path.join(ROOT, "tests", "data", "smarts", "extra_matches.json")

PATTERNS = [
    "[!#1]~[!#1]~[!#1]",
    "[R2]",
    "[r5,r6]@[r{5-6}]",
    "[x3]~[x2]",
    "[$(c1ccccc1)]-,:[!#1]",
    "[$([#6]=[#8])]~[!#1]",
    "[C;!R]-[N,O]",
    "*1~*~*~*~*~1",
    "*1~*~*~*~*~*~1",
    "[#6]1~[#6]~[#6]1",
    "c:c-[!#1]",
    "[CX4H2][CX4H2]",
    "[#6;^3]-[#6;^2]",
    "[N+,O-]",
    "[#7;H1,H2]",
    "[D3;R]",
    "[v4;!a]~[#1]",
    "[a;r6]1:[a]:[a]:[a]:[a]:[a]1",
    "O=C-N",
    "[!R;D2;!#1]",
    "[C@H]",
    "[k5]",
    "[k{6-}]",
    "[z2]",
    "[Z1]",
    "[*;R]!@[*;R]",
    "[C]1[C][C]1.[O]",
    "[$([C;$(C=O)])]O",
    "[c,n;H1]",
    "[!$(*=*);C;!$(C-[#1])]",
    "[X2]#*",
    "C~C(~O)~O",
    "[CH3]C(=O)[O,N]",
    "[r{-5}]@[r{7-}]",
    "[#6]-&!@[#6]",
    "[#8,#7;!H0]",
    "[R1]@[R1]@[R2]",
    "[d3]",
    "[X4;R]-[X4;R]",
    "[$(N(C)C=O),$(OC=O)]",
    "[CH2]2[CH2][CH2]1[CH2][CH2]2[CH2]1",
    "[#6]%11~[#6]~[#6]~[#6]%11",
    "[F,Cl,Br]-c",
    "[#6]=,#[#6]",
    "[!#6;!#1]",
    "[S;D3,D4]",
]


def main():
    suppl = Chem.SDMolSupplier(SDF, removeHs=False)
    mols = [m for m in suppl]
    out = {"patterns": [], "matches": {}}
    qs = []
    for p in PATTERNS:
        q = Chem.MolFromSmarts(p)
        if q is None:
            raise RuntimeError(p)
        qs.append(q)
        out["patterns"].append({
            "smarts": p,
            "num_atoms": q.GetNumAtoms(),
            "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in q.GetBonds()],
        })
    for m in mols:
        name = m.GetProp("_Name")
        per = {}
        for k, q in enumerate(qs):
            a = m.GetSubstructMatches(q, uniquify=False, useChirality=False, maxMatches=100000)
            if not a:
                continue
            u = m.GetSubstructMatches(q, uniquify=True, maxMatches=100000)
            per[str(k)] = {"all": [list(x) for x in a], "unique": [list(x) for x in u]}
        out["matches"][name] = per
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
