#!/usr/bin/env python
"""Fixtures for tests/assembly_search.rs (RDKit 2026.09.1).

Usage:
    py tools/gen_assembly_reference.py monomers [out dir]
    py tools/gen_assembly_reference.py reference [dir]

``monomers`` writes ``water.sdf``, ``acetic_acid.sdf`` and ``benzene.sdf``
(explicit hydrogens, ETKDGv3 seed 42, then MMFFOptimizeMolecule) into
``tests/data/assembly``.

``reference`` reads every ``*_found.sdf`` in the directory (written by the
ignored test ``write_assembly_fixtures`` in tests/assembly_search.rs: the
lowest structures found by our assembly search) and, for each record,
re-minimizes it with RDKit MMFF94 with ``ignoreInterfragInteractions=False``
starting from our final coordinates. It writes ``<name>_rdkit.ref``::

    REC <index>
    E0 <RDKit MMFF energy at our coordinates>
    E1 <energy after ff.Minimize(maxIts=10000)> <minimize return code>
    X <3N coordinates after the RDKit minimization>

Floats are written with repr() (round-trip exact).
"""
import glob
import os
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers as FFH

MONOMERS = [("water", "O"), ("acetic_acid", "CC(=O)O"), ("benzene", "c1ccccc1")]


def monomers(out):
    os.makedirs(out, exist_ok=True)
    for name, smi in MONOMERS:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        assert AllChem.EmbedMolecule(m, p) == 0
        assert FFH.MMFFOptimizeMolecule(m, maxIters=10000) == 0
        m.SetProp("_Name", name)
        w = Chem.SDWriter(os.path.join(out, name + ".sdf"))
        w.write(m)
        w.close()


def reference(d):
    for path in sorted(glob.glob(os.path.join(d, "*_found.sdf"))):
        name = os.path.basename(path)[: -len("_found.sdf")]
        lines = []
        for k, m in enumerate(Chem.SDMolSupplier(path, removeHs=False, sanitize=True)):
            props = FFH.MMFFGetMoleculeProperties(m)
            ff = FFH.MMFFGetMoleculeForceField(m, props, ignoreInterfragInteractions=False)
            e0 = ff.CalcEnergy()
            rc = ff.Minimize(maxIts=10000)
            e1 = ff.CalcEnergy()
            pos = ff.Positions()
            lines.append("REC %d" % k)
            lines.append("E0 %r" % e0)
            lines.append("E1 %r %d" % (e1, rc))
            lines.append("X " + " ".join(repr(float(x)) for x in pos))
        with open(os.path.join(d, name + "_rdkit.ref"), "w", newline="\n") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data", "assembly")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "monomers"
    d = sys.argv[2] if len(sys.argv) > 2 else here
    if cmd == "monomers":
        monomers(d)
    elif cmd == "reference":
        reference(d)
    else:
        sys.exit("unknown command " + cmd)
