"""Generates the benchmark molecules in this directory (RDKit 2026.09.1).

    py benches/data/make_bench_mols.py [200] [2000]

* peptide_200.mol  : 18-residue peptide, explicit H, 3D (282 atoms);
  ETKDGv3 embedding (random-coordinate start) + 200 MMFF94 steps
* peptide_2000.mol : 124-residue peptide, explicit H, 3D (~2000 atoms,
  V3000). ETKDG does not finish in reasonable time for this size, so the
  geometry is an expanded 2D layout with random z offsets, relaxed with
  500 MMFF94 steps (a compact, clash-free 3D structure; adequate for
  timing, not a realistic fold).
"""

import os
import random
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = "ACDEFGHIKLMNPQRSTVWY"


def peptide(n_res):
    seq = (SEQ * (n_res // len(SEQ) + 1))[:n_res]
    return seq, Chem.AddHs(Chem.MolFromSequence(seq))


def write(mol, name, seq):
    mol.SetProp("_Name", f"{name} ({seq})")
    block = Chem.MolToMolBlock(mol, forceV3000=mol.GetNumAtoms() > 999)
    with open(os.path.join(HERE, name + ".mol"), "w", newline="\n") as f:
        f.write(block)
    print(name, mol.GetNumAtoms(), "atoms", flush=True)


def make_small():
    seq, mol = peptide(18)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    ps.useRandomCoords = True
    if AllChem.EmbedMolecule(mol, ps) != 0:
        sys.exit("embedding failed for peptide_200")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    write(mol, "peptide_200", seq)


def make_large():
    seq, mol = peptide(124)
    AllChem.Compute2DCoords(mol)
    rng = random.Random(7)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x * 1.5, p.y * 1.5, rng.uniform(-1.5, 1.5)))
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    write(mol, "peptide_2000", seq)


if __name__ == "__main__":
    which = sys.argv[1:] or ["200", "2000"]
    if "200" in which:
        make_small()
    if "2000" in which:
        make_large()
