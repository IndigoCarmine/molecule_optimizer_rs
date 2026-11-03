"""Read tests/data/mol2/bench_initial.mol2 (written by our MOL2 writer) with
RDKit and record MMFF94 / UFF energies -> tests/data/mol2/rdkit_energies.tsv.

Usage: py tools/gen_mol2_reference.py
"""
import os
from rdkit import Chem
from rdkit.Chem import AllChem

d = os.path.join("tests", "data", "mol2")
text = open(os.path.join(d, "bench_initial.mol2")).read()
blocks = ["@<TRIPOS>MOLECULE" + b for b in text.split("@<TRIPOS>MOLECULE")[1:]]
rows = ["index\tname\tff\tenergy"]
for i, b in enumerate(blocks):
    m = Chem.MolFromMol2Block(b, removeHs=False)
    name = b.splitlines()[1].strip()
    if m is None:
        rows += [f"{i}\t{name}\tMMFF\tread_failed", f"{i}\t{name}\tUFF\tread_failed"]
        continue
    # MMFFGetMoleculeProperties modifies the molecule (MMFF aromaticity), so
    # every force field gets its own copy.
    mm = Chem.Mol(m)
    p = AllChem.MMFFGetMoleculeProperties(mm)
    if p is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mm, p)
        rows.append(f"{i}\t{name}\tMMFF\t{ff.CalcEnergy(ff.Positions())!r}")
    else:
        rows.append(f"{i}\t{name}\tMMFF\tno_params")
    mu = Chem.Mol(m)
    if AllChem.UFFHasAllMoleculeParams(mu):
        ff = AllChem.UFFGetMoleculeForceField(mu)
        rows.append(f"{i}\t{name}\tUFF\t{ff.CalcEnergy(ff.Positions())!r}")
with open(os.path.join(d, "rdkit_energies.tsv"), "w", newline="\n") as f:
    f.write("\n".join(rows) + "\n")
print(len(blocks), "molecules")
