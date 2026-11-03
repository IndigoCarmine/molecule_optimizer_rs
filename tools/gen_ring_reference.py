#!/usr/bin/env python
"""Generate RDKit ring-perception reference fixtures for tests/ring_reference.rs.

Usage:
    py tools/gen_ring_reference.py

Writes into ``tests/data/rings/``:

* ``tricky.sdf``  - hand-picked hard ring systems (cages, fullerenes,
  cyclophanes, spiro/bridged/fused systems, macrocycles, porphyrin, ...)
  built from SMILES with explicit hydrogens and 3D-free (2D) coordinates.
* ``rings.json``  - for every molecule of the input files below, RDKit's
  ``GetRingInfo().AtomRings()`` and ``BondRings()`` exactly as RDKit returns
  them (same ring order, same start atom and direction inside each ring)::

      {"<repo-relative sdf path>": [
          {"index": <record index>, "name": <title>,
           "atom_rings": [[...], ...], "bond_rings": [[...], ...],
           # aromatic molecules only: canonical atom ranks
           # (CanonicalRankAtomsInFragment over all atoms/bonds) and the
           # double bonds after the canonical Kekulize MMFF typing uses
           "ranks": [...], "kekule_double": [...]}
          | {"index": <record index>, "name": <title>, "failed": true},
          ...]}

Inputs: all of ``benches/data/set/initial.sdf``, every 4th record of
``tests/data/mmff/*.sdf``, all of ``tests/data/etkdg/molecules.sdf`` and
``tests/data/rings/tricky.sdf``.  Molecules are read with
``SDMolSupplier(removeHs=False)`` (default sanitization), i.e. exactly the way
the other reference generators read them.

Note: RDKit's default ring finding (2026.09, ``sanitizeMol`` ->
``symmetrizeSSSR``) returns RingDecomposerLib's relevant cycles in the order
of its relevant-cycle iterator, each ring normalized to start at its smallest
atom index; see src/perception/ring_decomposer.rs.

Requires the RDKit Python bindings (tested with 2026.09.1).
"""
import json
import os
import sys

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "data", "rings")

TRICKY = [
    ("cubane", "C12C3C4C1C5C2C3C45"),
    ("adamantane", "C1C2CC3CC1CC(C2)C3"),
    ("diamantane", "C1C2CC3C4CC5CC(C4)C(C2)C3C15"),
    ("prismane", "C12C3C1C1C2C31"),
    ("tetrahedrane", "C12C3C1C23"),
    ("bicyclobutane", "C1C2CC12"),
    ("propellane_111", "C1C23CC12C3"),
    ("norbornane", "C1CC2CCC1C2"),
    ("bicyclo222octane", "C1CC2CCC1CC2"),
    ("bicyclo333undecane", "C1CC2CCCC(C1)CCC2"),
    ("twistane", "C1CC2CC3CC1CC2C3"),
    ("triptycene", "c1ccc2c(c1)C1c3ccccc3C2c2ccccc21"),
    ("spiro44nonane", "C1CCC2(C1)CCCC2"),
    ("spiro45decane", "C1CCC2(CC1)CCCC2"),
    ("dispiro", "C1CC12CC21CC1"),
    ("paracyclophane_22", "C1Cc2ccc(cc2)CCc2ccc1cc2"),
    ("metacyclophane_22", "C1Cc2cccc(c2)CCc2cccc1c2"),
    ("cyclododecane", "C1CCCCCCCCCCC1"),
    ("crown18_6", "C1COCCOCCOCCOCCOCCO1"),
    ("cryptand222", "C1COCCOCCN2CCOCCOCCN1CCOCCOCC2"),
    ("porphine", "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3"),
    ("coronene", "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61"),
    ("helicene6", "c1ccc2c(c1)ccc1ccc3ccc4ccc5ccccc5c4c3c12"),
    ("cholesterol", "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"),
    ("morphine", "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"),
    ("strychnine", "C1CN2CC3=CCOC4CC(=O)N5C6C4C3CC2C61C7=CC=CC=C75"),
    ("quinuclidine", "C1CN2CCC1CC2"),
    ("dabco", "C1CN2CCN1CC2"),
    ("hexamethylenetetramine", "C1N2CN3CN1CN(C2)C3"),
    ("basketane", "C1CC2C3C4C1C5C2C3C45"),
    ("cuneane", "C12C3C4C1C1C2C3C41"),
    ("bullvalene", "C1=CC2C3C2C=CC13C=C"),
    ("pagodane_like", "C12C3C4C5C1C1C6C2C2C3C3C4C4C5C1C6C243"),
    ("dodecahedrane", "C12C3C4C5C1C1C6C7C2C2C8C3C3C9C4C4C5C5C1C1C6C6C7C2C2C8C3C9C4C5C1C62"),
    ("fullerene_C60", "c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41"),
    ("catenane_like_two_rings", "C1CCCCCCCCC1.C1CCCCCCC1"),
    ("salt_with_rings", "c1ccccc1C(=O)[O-].[Na+].C1CC1"),
    ("calixarene4", "Oc1c2Cc3cccc(Cc4cccc(Cc5cccc(Cc1ccc2)c5O)c4O)c3O"),
    ("cyclodextrin_alpha", "OCC1OC2OC3C(CO)OC(OC4C(CO)OC(OC5C(CO)OC(OC6C(CO)OC(OC7C(CO)OC(OC1C(O)C2O)C(O)C7O)C(O)C6O)C(O)C5O)C(O)C4O)C(O)C3O"),
    ("taxol_core", "CC1=C2C(C(=O)C3(C)C(CC4OCC4(C3C(C(C2(C)C)(CC1O)O)OC(=O)C5=CC=CC=C5)OC(C)=O)O)OC(C)=O"),
    ("vitamin_b12_corrin_like", "CC1=C2N=C(C=C3N=C(C(C)=C4NC(C)(C5N=C1C(C)C5)C(C)C4)C(C)C3)C(C)C2"),
    ("bicyclo_fused_4_4", "C1CC2C1C1CCC21"),
    ("ladderane3", "C1CC2C1C1CC2C1"),
    ("pentacyclo_k4_like", "C12C3C4C1C2C34"),
    ("figueras_dup_example", "C12=CON=C1C(C4)CC3CC2CC4C3"),
    ("issue134", "SC(C3C1CC(C3)CC(C2S)(O)C1)2S"),
    ("tosyl_sugar", "CC1=CC=C(C=C1)S(=O)(=O)OC2C3COC(O3)C4OC(C)(C)OC24"),
]


def ring_record(idx, mol):
    if mol is None:
        return {"index": idx, "name": None, "failed": True}
    ri = mol.GetRingInfo()
    rec = {
        "index": idx,
        "name": mol.GetProp("_Name") if mol.HasProp("_Name") else "",
        "atom_rings": [list(r) for r in ri.AtomRings()],
        "bond_rings": [list(r) for r in ri.BondRings()],
    }
    if any(a.GetIsAromatic() for a in mol.GetAtoms()):
        n, nb = mol.GetNumAtoms(), mol.GetNumBonds()
        rec["ranks"] = list(Chem.CanonicalRankAtomsInFragment(mol, list(range(n)), list(range(nb))))
        km = Chem.RWMol(mol)
        try:
            Chem.Kekulize(km, clearAromaticFlags=True)  # canonical=True
            rec["kekule_double"] = [b.GetIdx() for b in km.GetBonds()
                                    if b.GetBondType() == Chem.BondType.DOUBLE]
        except Exception:
            rec["kekule_double"] = None
    return rec


def write_tricky(path):
    blocks = []
    for name, smi in TRICKY:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print(f"skipping invalid SMILES {name}", file=sys.stderr)
            continue
        m = Chem.AddHs(m)
        AllChem.Compute2DCoords(m)
        m.SetProp("_Name", name)
        blocks.append(Chem.MolToMolBlock(m) + "$$$$\n")
    with open(path, "w", newline="\n") as fh:
        fh.write("".join(blocks))
    print(f"tricky.sdf: {len(blocks)} molecules")


def main():
    print("RDKit", rdBase.rdkitVersion)
    os.makedirs(OUT, exist_ok=True)
    write_tricky(os.path.join(OUT, "tricky.sdf"))

    inputs = [("benches/data/set/initial.sdf", 1)]
    mmff_dir = os.path.join(ROOT, "tests", "data", "mmff")
    for f in sorted(os.listdir(mmff_dir)):
        if f.endswith(".sdf"):
            inputs.append((f"tests/data/mmff/{f}", 4))
    inputs.append(("tests/data/etkdg/molecules.sdf", 1))
    inputs.append(("tests/data/rings/tricky.sdf", 1))

    result = {}
    for rel, step in inputs:
        supp = Chem.SDMolSupplier(os.path.join(ROOT, rel), removeHs=False)
        recs = []
        for i in range(0, len(supp), step):
            recs.append(ring_record(i, supp[i]))
        result[rel] = recs
        nr = sum(len(r.get("atom_rings", [])) for r in recs)
        print(f"{rel}: {len(recs)} molecules, {nr} rings")

    with open(os.path.join(OUT, "rings.json"), "w", newline="\n") as fh:
        json.dump(result, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
