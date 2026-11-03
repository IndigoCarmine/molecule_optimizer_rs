#!/usr/bin/env python
"""Generate RDKit MMFF94/MMFF94s reference fixtures for tests/mmff_reference.rs.

Usage:
    py tools/gen_mmff_reference.py <rdkit MMFF test_data dir> [out dir]

For each validation set (MMFF94_dative, MMFF94_hypervalent, MMFF94s_dative,
MMFF94s_hypervalent) this writes

* ``<set>.sdf``  - the input molecules (atom/bond lines with trailing all-zero
  columns trimmed; coordinates untouched).  The reference data below is
  computed from *this* file, so the Rust tests read exactly the same input.
* ``<set>.ref``  - line based reference data, one block per molecule::

      MOL <name> <natoms> <status>         status: ok | noparams | readfail
      T <MMFF atom type per atom>
      Q <MMFF partial charge per atom>
      E <total> <bond> <angle> <stbn> <oop> <torsion> <vdw> <ele>
      X <E dist-dielectric eps=4> <E ignoreInterfrag=False> <E nonBondedThresh=9>
      G <gradient, 3N values>
      O10 <status> <energy> <3N coordinates after Minimize(maxIts=10)>
      O200 <status> <energy> <3N coordinates after MMFFOptimizeMolecule()>
      P <3N perturbed start coordinates (exact 4-decimal values)>
      PE <energy at the perturbed coordinates>
      PO200 <status> <energy> <3N coordinates after minimising from P>

  Energies and gradients are written with repr() (round-trip exact).
  If the hypervalent set gives byte-identical results to the dative set, its
  .ref file is omitted and the tests reuse the dative one.

Requires the RDKit Python bindings (tested with 2026.09.1).
"""
import os
import random
import sys

from rdkit import Chem, rdBase
from rdkit.Chem import rdForceFieldHelpers as FFH
from rdkit.Geometry import Point3D

PERTURB = 0.15  # max coordinate perturbation (Angstrom) for the P/PO200 runs

SETS = [
    ("MMFF94_dative", "MMFF94"),
    ("MMFF94_hypervalent", "MMFF94"),
    ("MMFF94s_dative", "MMFF94s"),
    ("MMFF94s_hypervalent", "MMFF94s"),
]

TERMS = ["Bond", "Angle", "StretchBend", "Oop", "Torsion", "VdW", "Ele"]


def trim_sdf(text):
    """Strip trailing all-zero columns from V2000 atom and bond lines."""
    out = []
    state = 0  # 0 header, 1 atoms, 2 bonds, 3 rest
    na = nb = k = 0
    hdr = 0
    for line in text.splitlines():
        if state == 0:
            hdr += 1
            if hdr == 4:
                na = int(line[0:3]); nb = int(line[3:6]); k = 0
                state = 1 if na else (2 if nb else 3)
            out.append(line)
            continue
        if state == 1:
            fixed, rest = line[:39], line[39:]
            if rest.split() and all(x == "0" for x in rest.split()):
                line = fixed
            k += 1
            if k == na:
                state = 2 if nb else 3
                k = 0
        elif state == 2:
            fixed, rest = line[:9], line[9:]
            if rest.split() and all(x == "0" for x in rest.split()):
                line = fixed
            k += 1
            if k == nb:
                state = 3
        if line.startswith("$$$$"):
            state = 0; hdr = 0
        out.append(line)
    return "\n".join(out) + "\n"


def fmt(xs, f):
    return " ".join(f % x for x in xs)


def final_energy(ff):
    """Energy at the force field's current positions.

    ForceField.CalcEnergy() without arguments reuses RDKit's internal
    distance cache, which still holds the distances of the last line-search
    trial point when the final line search failed (the positions themselves
    are reset to the previous point). Passing the positions explicitly
    resets the cache and gives the true energy of the returned geometry.
    """
    return ff.CalcEnergy(list(ff.Positions()))


def energy_with(mol, variant, **kw):
    props = FFH.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    if "only" in kw:
        for t in TERMS:
            getattr(props, "SetMMFF%sTerm" % t)(t == kw["only"])
    ff = FFH.MMFFGetMoleculeForceField(
        mol, props,
        nonBondedThresh=kw.get("thresh", 100.0),
        ignoreInterfragInteractions=kw.get("ignore", True))
    return ff.CalcEnergy()


def main():
    src = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data", "mmff")
    os.makedirs(out_dir, exist_ok=True)
    rdBase.DisableLog("rdApp.*")
    for name, variant in SETS:
        with open(os.path.join(src, name + ".sdf"), newline="") as fh:
            text = trim_sdf(fh.read())
        sdf_out = os.path.join(out_dir, name + ".sdf")
        with open(sdf_out, "w", newline="\n") as fh:
            fh.write(text)
        lines = []
        nok = 0
        supp = Chem.SDMolSupplier(sdf_out, removeHs=False)
        for idx in range(len(supp)):
            # MMFFMolProperties modifies the molecule it is built from
            # (kekulization + MMFF aromaticity), so every calculation below
            # starts from a freshly parsed molecule, as a user would.
            def fresh():
                return supp[idx]

            mol = fresh()
            if mol is None:
                lines.append("MOL #%d 0 readfail" % idx)
                continue
            n = mol.GetNumAtoms()
            mname = mol.GetProp("_Name").strip() or ("#%d" % idx)
            if not FFH.MMFFHasAllMoleculeParams(mol):
                lines.append("MOL %s %d noparams" % (mname, n))
                continue
            mol = fresh()
            props = FFH.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
            lines.append("MOL %s %d ok" % (mname, n))
            lines.append("T " + " ".join(str(props.GetMMFFAtomType(i)) for i in range(n)))
            lines.append("Q " + fmt([props.GetMMFFPartialCharge(i) for i in range(n)], "%.10f"))
            ff = FFH.MMFFGetMoleculeForceField(mol, props)
            e = ff.CalcEnergy()
            terms = [energy_with(fresh(), variant, only=t) for t in TERMS]
            lines.append("E %r " % e + fmt(terms, "%r"))
            # extra settings
            m2 = fresh()
            p2 = FFH.MMFFGetMoleculeProperties(m2, mmffVariant=variant)
            p2.SetMMFFDielectricModel(2)  # 1 = constant, 2 = distance-dependent
            p2.SetMMFFDielectricConstant(4.0)
            e_dist = FFH.MMFFGetMoleculeForceField(m2, p2).CalcEnergy()
            m3 = fresh()
            e_inter = FFH.MMFFGetMoleculeForceField(
                m3, FFH.MMFFGetMoleculeProperties(m3, mmffVariant=variant),
                ignoreInterfragInteractions=False).CalcEnergy()
            m4 = fresh()
            e_th = FFH.MMFFGetMoleculeForceField(
                m4, FFH.MMFFGetMoleculeProperties(m4, mmffVariant=variant),
                nonBondedThresh=9.0).CalcEnergy()
            lines.append("X %r %r %r" % (e_dist, e_inter, e_th))
            lines.append("G " + fmt(ff.CalcGrad(), "%r"))
            # short trajectory and default optimisation (== MMFFOptimizeMolecule)
            for tag, its, cf in (("O10", 10, "%.9f"), ("O200", 200, "%.7f")):
                mo = fresh()
                ffo = FFH.MMFFGetMoleculeForceField(
                    mo, FFH.MMFFGetMoleculeProperties(mo, mmffVariant=variant))
                ffo.Initialize()
                st = ffo.Minimize(maxIts=its)
                lines.append("%s %d %r " % (tag, st, final_energy(ffo)) + fmt(ffo.Positions(), cf))
            # the validation-suite geometries are already close to their MMFF
            # minima: also minimise from a reproducibly perturbed start
            # (coordinates written with 4 decimals and read back exactly)
            rng = random.Random(1000003 * (idx + 1) + len(variant))
            mo = fresh()
            conf = mo.GetConformer()
            pert = []
            for i in range(n):
                p = conf.GetAtomPosition(i)
                q = [float("%.4f" % (c + rng.uniform(-PERTURB, PERTURB))) for c in (p.x, p.y, p.z)]
                conf.SetAtomPosition(i, Point3D(*q))
                pert.extend(q)
            lines.append("P " + fmt(pert, "%.4f"))
            ffo = FFH.MMFFGetMoleculeForceField(
                mo, FFH.MMFFGetMoleculeProperties(mo, mmffVariant=variant))
            ffo.Initialize()
            lines.append("PE %r" % ffo.CalcEnergy())
            st = ffo.Minimize(maxIts=200)
            lines.append("PO200 %d %r " % (st, final_energy(ffo)) + fmt(ffo.Positions(), "%.7f"))
            nok += 1
        text = "\n".join(lines) + "\n"
        ref_out = os.path.join(out_dir, name + ".ref")
        base = os.path.join(out_dir, name.replace("hypervalent", "dative") + ".ref")
        if name.endswith("hypervalent") and open(base).read() == text:
            # RDKit gives bit-identical results for both representations;
            # the tests then reuse the dative reference file.
            if os.path.exists(ref_out):
                os.remove(ref_out)
            print("%s: identical to %s" % (name, os.path.basename(base)))
        else:
            with open(ref_out, "w", newline="\n") as fh:
                fh.write(text)
        print("%s: %d molecules, %d with parameters" % (name, len(supp), nok))


if __name__ == "__main__":
    main()
