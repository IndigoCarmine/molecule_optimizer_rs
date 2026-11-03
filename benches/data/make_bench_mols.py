"""Generates the benchmark molecules in this directory (RDKit 2026.09.1).

    py benches/data/make_bench_mols.py

* peptide_200.mol  : 18-residue peptide, explicit H, 3D (~200 atoms)
* peptide_2000.mol : 180-residue peptide, explicit H, 3D (~2000 atoms, V3000)

Coordinates: ETKDG embedding (random-coordinate start; chirality not
enforced for the large peptide)
followed by a short MMFF94 relaxation so that the geometry is reasonable.
"""

import os
import sys

from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = "ACDEFGHIKLMNPQRSTVWY"


def make(n_res, name, seed):
    seq = (SEQ * (n_res // len(SEQ) + 1))[:n_res]
    mol = Chem.AddHs(Chem.MolFromSequence(seq))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.useRandomCoords = True
    if n_res > 50:
        # stereo is irrelevant for timing; enforcing it makes the embedding
        # of the large peptide retry for a very long time
        ps.enforceChirality = False
        ps.maxIterations = 50
    if AllChem.EmbedMolecule(mol, ps) != 0:
        sys.exit(f"embedding failed for {name}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    mol.SetProp("_Name", f"{name} ({seq})")
    block = Chem.MolToMolBlock(mol, forceV3000=mol.GetNumAtoms() > 999)
    with open(os.path.join(HERE, name + ".mol"), "w", newline="\n") as f:
        f.write(block)
    print(name, mol.GetNumAtoms(), "atoms", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["200", "2000"]
    if "200" in which:
        make(18, "peptide_200", 42)
    if "2000" in which:
        make(180, "peptide_2000", 7)
