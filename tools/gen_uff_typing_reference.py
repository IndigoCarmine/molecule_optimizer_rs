"""Generates RDKit reference data for the UFF atom typing used by the
distance-geometry bounds-matrix builder (src/distgeom/uff_params.rs).

Usage:
    py tools/gen_uff_typing_reference.py [extra]

Outputs:
    tests/data/etkdg/uff_typing.json        for tests/data/etkdg/molecules.sdf
    tests/data/etkdg/uff_typing_extra.sdf   extra molecules (hypervalent S/P,
                                            charged N/O, B, Si, halogens, Se);
                                            only regenerated with `extra`
    tests/data/etkdg/uff_typing_extra.json  reference data for those

For every molecule (read with removeHs=False) the JSON records
    has_all : rdForceFieldHelpers.UFFHasAllMoleculeParams
    bonds   : [begin, end, r0 | null]  (GetUFFBondStretchParams)
    theta0  : per atom, theta0 in degrees of the first angle centred on it
              (GetUFFAngleBendParams) or null
    labels  : per atom, the UFF label inferred from r0 / theta0 against the
              UFF parameter table (RDKit does not expose getAtomLabel to
              Python); null where the data does not single out one label
              (e.g. isolated atoms), or where RDKit has no parameters.

Requires RDKit (tested with 2026.09.1).
"""
import json
import math
import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "tests", "data", "etkdg")
TABLE = os.path.join(ROOT, "src", "distgeom", "data", "uff_params.txt")

EXTRA = [
    ("dmso", "CS(=O)C"),
    ("sulfone", "CS(=O)(=O)C"),
    ("sulfonamide", "CS(=O)(=O)N"),
    ("sulfuric_acid", "OS(=O)(=O)O"),
    ("thiophene", "c1ccsc1"),
    ("thioacetone", "CC(=S)C"),
    ("thiol", "CCS"),
    ("phosphine", "CP(C)C"),
    ("phosphate", "COP(=O)(OC)OC"),
    ("phosphonium", "C[P+](C)(C)C"),
    ("ammonium", "C[N+](C)(C)C"),
    ("nitro", "C[N+](=O)[O-]"),
    ("carboxylate", "CC(=O)[O-]"),
    ("pyridinium", "c1cc[nH+]cc1"),
    ("oxonium", "C[OH2+]"),
    ("borane", "CB(C)C"),
    ("boronic_acid", "c1ccccc1B(O)O"),
    ("silane", "C[Si](C)(C)C"),
    ("chlorobenzene", "Clc1ccccc1"),
    ("bromoethane", "CCBr"),
    ("iodomethane", "CI"),
    ("trifluoromethyl", "FC(F)(F)c1ccccc1"),
    ("selenol", "CC[SeH]"),
    ("selenophene", "c1cc[se]c1"),
    ("nitrile", "CC#N"),
    ("isocyanate", "CN=C=O"),
    ("azide", "CN=[N+]=[N-]"),
    ("amide", "CC(=O)NC"),
    ("furan", "c1ccoc1"),
    ("enol_ether", "C=COC"),
]


def read_table():
    tab = {}
    for line in open(TABLE):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        tab[f[0]] = (float(f[1]), float(f[2]), float(f[9]))  # r1, theta0, Xi
    return tab


def rest_length(bo, a, b):
    ri, rj = a[0], b[0]
    xi, xj = a[2], b[2]
    rbo = -0.1332 * (ri + rj) * math.log(bo)
    ren = ri * rj * (math.sqrt(xi) - math.sqrt(xj)) ** 2 / (xi * ri + xj * rj)
    return ri + rj + rbo - ren


def element_of_label(lab):
    return lab[:2].rstrip("_0123456789+")


def infer_labels(m, bonds, theta0, tab):
    cands = []
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        c = [l for l in tab if element_of_label(l) == sym]
        t = theta0[a.GetIdx()]
        if t is not None:
            c = [l for l in c if abs(tab[l][1] - t) < 1e-6]
        cands.append(set(c))
    changed = True
    while changed:
        changed = False
        for (i, j, r0), b in zip(bonds, m.GetBonds()):
            if r0 is None:
                continue
            bo = b.GetBondTypeAsDouble()
            for x, y in ((i, j), (j, i)):
                keep = {l for l in cands[x]
                        if any(abs(rest_length(bo, tab[l], tab[k]) - r0) < 1e-9 for k in cands[y])}
                if keep != cands[x]:
                    cands[x] = keep
                    changed = True
    constrained = [t is not None for t in theta0]
    for i, j, r0 in bonds:
        if r0 is not None:
            constrained[i] = constrained[j] = True
    out = []
    for a, c in zip(m.GetAtoms(), cands):
        if not constrained[a.GetIdx()]:
            out.append(None)
            continue
        # labels with identical r1/theta0/Xi cannot be told apart; prefer
        # the one the typer can actually produce (P_3+5 over P_3+q).
        c = {l for l in c if l not in ("P_3+q", "O_3_z")} or c
        out.append(next(iter(c)) if len(c) == 1 else None)
    return out


def reference(m, tab):
    n = m.GetNumAtoms()
    bonds = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        p = rdForceFieldHelpers.GetUFFBondStretchParams(m, i, j)
        bonds.append([i, j, p[1] if p else None])
    theta0 = [None] * n
    for a in m.GetAtoms():
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        if len(nb) >= 2:
            p = rdForceFieldHelpers.GetUFFAngleBendParams(m, nb[0], a.GetIdx(), nb[1])
            if p:
                theta0[a.GetIdx()] = p[1]
    labels = infer_labels(m, bonds, theta0, tab)
    return {
        "name": m.GetProp("_Name"),
        "has_all": bool(rdForceFieldHelpers.UFFHasAllMoleculeParams(m)),
        "bonds": bonds,
        "theta0": theta0,
        "labels": labels,
    }


def write_extra_sdf(path):
    w = Chem.SDWriter(path)
    w.SetForceV3000(False)
    for name, smi in EXTRA:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = 1234
        if rdDistGeom.EmbedMolecule(m, ps) < 0:
            raise RuntimeError("embedding failed for " + name)
        m.SetProp("_Name", name)
        m.SetProp("SMILES", smi)
        w.write(m)
    w.close()


def process(sdf, out_json, tab):
    res = []
    for m in Chem.SDMolSupplier(sdf, removeHs=False):
        if m is None:
            raise RuntimeError("could not read molecule from " + sdf)
        res.append(reference(m, tab))
    with open(out_json, "w", newline="\n") as f:
        json.dump(res, f, indent=1)
    unresolved = sum(l is None for r in res for l in r["labels"])
    print(f"wrote {out_json}: {len(res)} molecules, {unresolved} unresolved labels")


def main(argv):
    RDLogger.DisableLog("rdApp.*")
    tab = read_table()
    extra_sdf = os.path.join(OUT, "uff_typing_extra.sdf")
    if "extra" in argv or not os.path.exists(extra_sdf):
        write_extra_sdf(extra_sdf)
    process(os.path.join(OUT, "molecules.sdf"), os.path.join(OUT, "uff_typing.json"), tab)
    process(extra_sdf, os.path.join(OUT, "uff_typing_extra.json"), tab)


if __name__ == "__main__":
    main(sys.argv[1:])
