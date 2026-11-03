#!/usr/bin/env python
"""Generates UFF reference fixtures from RDKit for `tests/uff_reference.rs`.

Usage:
    py tools/gen_uff_reference.py [RDKIT_SOURCE_DIR]

RDKIT_SOURCE_DIR is an RDKit source checkout (only the test data under
``Code/ForceField/MMFF/test_data`` and ``Code/GraphMol/ForceFieldHelpers/UFF/
test_data`` is read).  Defaults to the ``RDKIT_SRC`` environment variable.

For every molecule the script writes the molblock that RDKit reads (with
``removeHs=False``) together with the exact double-precision input
coordinates, the UFF atom types, the energy and gradient at the input
coordinates, and the result of ``UFFOptimizeMolecule`` after 10 and 200
iterations.  Output: ``tests/data/uff/uff_reference.txt``.

The atom labels are computed with a line-by-line Python port of RDKit's
``UFF::Tools::getAtomLabel`` (RDKit does not expose it) evaluated on the
RDKit-perceived molecule; they are cross-checked against
``GetUFFBondStretchParams`` / ``GetUFFAngleBendParams`` /
``UFFHasAllMoleculeParams``.
"""

import math
import os
import sys

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem import rdForceFieldHelpers as FFH

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "tests", "data", "uff", "uff_reference.txt")
PARAMS = os.path.join(ROOT, "src", "data", "uff_params.tsv")

PT = Chem.GetPeriodicTable()


def load_params():
    p = {}
    for line in open(PARAMS):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        p[f[0]] = [float(x) for x in f[1:]]
    return p


UFF_PARAMS = load_params()


# ---------------------------------------------------------------------------
# Port of RDKit Code/GraphMol/ForceFieldHelpers/UFF/AtomTyper.cpp
# (getAtomLabel / addAtomChargeFlags, tolerateChargeMismatch = true)
# ---------------------------------------------------------------------------
def add_charge_flags(atom, key):
    tv = atom.GetTotalValence()
    fc = atom.GetFormalCharge()
    n = atom.GetAtomicNum()
    tol = True
    fixed = {
        29: 1, 47: 1,
        4: 2, 20: 2, 25: 2, 26: 2, 28: 2, 46: 2, 78: 2,
        21: 3, 24: 3, 27: 3, 79: 3, 89: 3, 96: 3, 97: 3, 98: 3, 99: 3,
        100: 3, 101: 3, 102: 3, 103: 3,
        2: 4, 18: 4, 22: 4, 36: 4, 54: 4, 90: 4, 91: 4, 92: 4, 93: 4, 94: 4, 95: 4,
        23: 5, 41: 5, 43: 5, 73: 5,
        42: 6,
    }
    if n in fixed:
        v = fixed[n]
        if tv == v or fc == v or tol:
            key += "+%d" % v
    elif n == 12:
        key += "+2"
    elif n == 15:
        key += "+3" if tv == 3 else "+5"
    elif n == 16:
        if atom.GetHybridization() != Chem.HybridizationType.SP2:
            key += {2: "+2", 4: "+4"}.get(tv, "+6")
    elif n in (30, 34, 48, 52, 80, 84):
        key += "+2"
    elif n in (31, 33, 49, 51, 81, 82, 83):
        key += "+3"
    elif n == 75:
        if key == "Re6":
            key = "Re6+5"
        elif key == "Re3":
            key = "Re3+7"
    if 57 <= n <= 71:
        key += "+3"
    return key


def atom_label(atom):
    n = atom.GetAtomicNum()
    key = atom.GetSymbol()
    if len(key) == 1:
        key += "_"
    if n:
        if PT.GetDefaultValence(n) == -1 or (
            PT.GetNOuterElecs(n) != 1 and PT.GetNOuterElecs(n) != 7
        ):
            if n in (12, 13, 14, 15, 50, 51, 52, 81, 82, 83, 84):
                key += "3"
            elif n == 80:
                key += "1"
            else:
                h = atom.GetHybridization()
                H = Chem.HybridizationType
                if h == H.S:
                    pass
                elif h == H.SP:
                    key += "1"
                elif h == H.SP2:
                    conj = any(b.GetIsConjugated() for b in atom.GetBonds())
                    if (atom.GetIsAromatic() or conj) and n in (6, 7, 8, 16):
                        key += "R"
                    else:
                        key += "2"
                elif h == H.SP3:
                    key += "3"
                elif h == H.SP2D:
                    key += "4"
                elif h == H.SP3D:
                    key += "5"
                elif h == H.SP3D2:
                    key += "6"
    return add_charge_flags(atom, key)


def bond_order(b):
    return b.GetBondTypeAsDouble()


def rest_length(bo, p1, p2):
    ri, rj = p1[0], p2[0]
    rbo = -0.1332 * (ri + rj) * math.log(bo)
    xi, xj = p1[8], p2[8]
    ren = ri * rj * (math.sqrt(xi) - math.sqrt(xj)) * (math.sqrt(xi) - math.sqrt(xj)) / (xi * ri + xj * rj)
    return ri + rj + rbo - ren


def check_labels(mol, labels):
    """Cross-checks the Python labels against RDKit's own typer."""
    known = [UFF_PARAMS.get(l) for l in labels]
    assert FFH.UFFHasAllMoleculeParams(mol) == all(k is not None for k in known), labels
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        res = FFH.GetUFFBondStretchParams(mol, i, j)
        if known[i] is None or known[j] is None:
            assert res is None, (labels[i], labels[j])
            continue
        if bond_order(b) <= 0:
            continue
        kb, r0 = res
        assert abs(r0 - rest_length(bond_order(b), known[i], known[j])) < 1e-9, (labels[i], labels[j])
    for a in mol.GetAtoms():
        j = a.GetIdx()
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        if len(nb) >= 2 and known[j] is not None and known[nb[0]] is not None and known[nb[1]] is not None:
            if all(bond_order(mol.GetBondBetweenAtoms(j, k)) > 0 for k in nb[:2]):
                res = FFH.GetUFFAngleBendParams(mol, nb[0], j, nb[1])
                assert abs(res[1] - known[j][1]) < 1e-9, labels[j]


# ---------------------------------------------------------------------------
# molecule sources
# ---------------------------------------------------------------------------
def sdf_records(path):
    text = open(path).read().replace("\r\n", "\n")
    recs = text.split("$$$$\n")
    out = []
    for r in recs:
        if "M  END" not in r:
            continue
        mb = r[: r.index("M  END") + len("M  END")] + "\n"
        out.append(mb)
    return out


SMILES = [
    # (name, smiles, add_hs)
    ("cyclopropene", "C1=CC1", True),
    ("methylenecyclopropane", "C=C1CC1", True),
    ("cyclobutene", "C1=CCC1", True),
    ("methylenecyclobutane", "C=C1CCC1", True),
    ("cyclopropanone", "O=C1CC1", True),
    ("acetonitrile", "CC#N", True),
    ("propyne", "CC#CC", True),
    ("allene", "C=C=C", True),
    ("co2", "O=C=O", True),
    ("formamide", "NC=O", True),
    ("nma", "CNC(C)=O", True),
    ("nitrobenzene", "c1ccccc1[N+](=O)[O-]", True),
    ("pyridinium", "c1cc[nH+]cc1", True),
    ("acetate", "CC(=O)[O-]", True),
    ("methylammonium", "C[NH3+]", True),
    ("sodium_acetate", "[Na+].CC(=O)[O-]", True),
    ("ethanol_water", "CCO.O", True),
    ("dmso", "CS(C)=O", True),
    ("sulfone", "CS(=O)(=O)C", True),
    ("sulfonamide", "CS(=O)(=O)N", True),
    ("methanesulfonic", "CS(=O)(=O)O", True),
    ("thiophene", "c1ccsc1", True),
    ("dimethyldisulfide", "CSSC", True),
    ("hydrogen_peroxide", "OO", True),
    ("dimethylperoxide", "COOC", True),
    ("methanethiol_ether", "CSOC", True),
    ("phosphate", "COP(=O)(OC)OC", True),
    ("phosphine", "CP(C)C", True),
    ("phosphonium", "C[P+](C)(C)C", True),
    ("pf5", "FP(F)(F)(F)F", True),
    ("pcl5", "ClP(Cl)(Cl)(Cl)Cl", True),
    ("sf6", "FS(F)(F)(F)(F)F", True),
    ("sf4", "FS(F)(F)F", True),
    ("tms", "C[Si](C)(C)C", True),
    ("germane", "C[Ge](C)(C)C", True),
    ("tin", "C[Sn](C)(C)C", True),
    ("lead", "C[Pb](C)(C)C", True),
    ("arsine", "C[As](C)C", True),
    ("stibine", "C[Sb](C)C", True),
    ("bismuthine", "C[Bi](C)C", True),
    ("selenide", "C[Se]C", True),
    ("telluride", "C[Te]C", True),
    ("selenophene", "c1cc[se]c1", True),
    ("borane", "CB(C)C", True),
    ("borate", "C[B-](C)(C)C", True),
    ("boronic", "OB(O)c1ccccc1", True),
    ("aluminium", "C[Al](C)C", True),
    ("bromo_iodo", "BrCCI", True),
    ("trifluoromethylbenzene", "FC(F)(F)c1ccccc1", True),
    ("azulene", "c1ccc2cccc2cc1", True),
    ("indole", "c1ccc2[nH]ccc2c1", True),
    ("imidazolium", "C[n+]1cc[nH]c1", True),
    ("furan", "c1ccoc1", True),
    ("oxazole", "c1cocn1", True),
    ("pyrimidinone", "O=c1cc[nH]c(=O)[nH]1", True),
    ("guanidinium", "NC(N)=[NH2+]", True),
    ("azide", "CN=[N+]=[N-]", True),
    ("diazo", "C=[N+]=[N-]", True),
    ("isocyanide", "C[N+]#[C-]", True),
    ("ketene", "C=C=O", True),
    ("carbodiimide", "CN=C=NC", True),
    ("oxime", "CC=NO", True),
    ("hydrazone", "CC=NN", True),
    ("enol_ether", "C=COC", True),
    ("enamine", "C=CN(C)C", True),
    ("acrolein", "C=CC=O", True),
    ("styrene", "C=Cc1ccccc1", True),
    ("biphenyl", "c1ccccc1-c1ccccc1", True),
    ("cubane", "C12C3C4C1C5C2C3C45", True),
    ("bicyclobutane", "C1C2C1C2", True),
    ("spiropentane", "C1CC12CC2", True),
    ("cyclooctatetraene", "C1=CC=CC=CC=C1", True),
    ("cyclopentadienyl_anion", "[cH-]1cccc1", True),
    ("tropylium", "c1cc[cH+]ccc1", True),
    ("zinc_complex", "C[Zn]C", True),
    ("cadmium", "C[Cd]C", True),
    ("mercury", "C[Hg]C", True),
    ("copper", "C[Cu]", True),
    ("silver", "[Ag]C#N", True),
    ("ferrocene_like", "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1", True),
    ("pt_square_planar", "Cl[Pt](Cl)([NH3])[NH3]", True),
    ("pd_complex", "Cl[Pd](Cl)(P)P", True),
    ("ni_tetracarbonyl", "O=C=[Ni](=C=O)(=C=O)=C=O", True),
    ("mo_hexacarbonyl", "O=C=[Mo](=C=O)(=C=O)(=C=O)(=C=O)=C=O", True),
    ("titanium", "Cl[Ti](Cl)(Cl)Cl", True),
    ("magnesium", "C[Mg]Br", True),
    ("lithium", "C[Li]", True),
    ("potassium_phenoxide", "[K+].[O-]c1ccccc1", True),
    ("calcium", "[Ca+2].[O-]C(=O)C.[O-]C(=O)C", True),
    ("co_complex", "[Co](N)(N)(N)(N)(N)N", True),
    ("xenon_difluoride", "F[Xe]F", True),
    ("radical_methyl", "[CH3]", False),
    ("carbene_like", "[CH2]", False),
    ("implicit_h_ethanol", "CCO", False),
    ("implicit_h_benzoic", "OC(=O)c1ccccc1", False),
    ("partial_h", "CC(O)C(=O)N", "partial"),
    ("dative_bond", "C[NH2]->[Zn](Cl)Cl", True),
    ("amine_oxide", "C[N+](C)(C)[O-]", True),
    ("sulfoxide_charge_sep", "C[S+](C)[O-]", True),
    ("nitrone", "C=[N+](C)[O-]", True),
    ("phosphine_oxide", "CP(C)(C)=O", True),
    ("phosphite", "COP(OC)OC", True),
    ("thioamide", "CC(N)=S", True),
    ("thiourea", "NC(N)=S", True),
    ("sulfonyl_chloride", "CS(=O)(=O)Cl", True),
    ("sulfate", "COS(=O)(=O)OC", True),
    ("big_peptide", "CC(C)C[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)O", True),
    ("steroid", "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc43)[C@@H]1CC[C@@H]2O", True),
    ("macrocycle", "C1CCCCCCCCCCC1", True),
    ("long_chain", "CCCCCCCCCCCCCCCCCCCC", True),
]


def embed(smi, add_hs, seed=0xF00D):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    if add_hs is True:
        m = Chem.AddHs(m)
    elif add_hs == "partial":
        # explicit Hs only on heteroatoms
        m = Chem.AddHs(m, onlyOnAtoms=[a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() != 6])
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.useRandomCoords = True
    if AllChem.EmbedMolecule(m, ps) != 0:
        return None
    return m


def fmt(x):
    return repr(float(x))


def pts(conf_or_list, n):
    if isinstance(conf_or_list, (list, tuple)):
        flat = list(conf_or_list)
        return [(flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]) for i in range(n)]
    return [tuple(conf_or_list.GetAtomPosition(i)) for i in range(n)]


def record(name, source, molblock, coords=None, vdw_thresh=10.0, interfrag=True):
    mol = Chem.MolFromMolBlock(molblock, removeHs=False)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    if n < 1:
        return None
    conf = mol.GetConformer()
    if coords is not None:
        for i, p in enumerate(coords):
            conf.SetAtomPosition(i, p)
    x0 = pts(conf, n)
    labels = [atom_label(a) for a in mol.GetAtoms()]
    check_labels(mol, labels)
    has_all = FFH.UFFHasAllMoleculeParams(mol)

    out = []
    out.append("$NAME %s" % name)
    out.append("$SOURCE %s" % source)
    out.append("$OPTS %s %d" % (fmt(vdw_thresh), 1 if interfrag else 0))
    out.append("$MOLBLOCK")
    out.append(molblock.rstrip("\n"))
    out.append("$COORDS %d" % n)
    for p in x0:
        out.append(" ".join(fmt(v) for v in p))
    out.append("$TYPES " + " ".join(labels))
    out.append("$HASALL %d" % (1 if has_all else 0))

    ff = FFH.UFFGetMoleculeForceField(
        mol, vdwThresh=vdw_thresh, ignoreInterfragInteractions=interfrag
    )
    if ff is None:
        out.append("$END")
        return "\n".join(out) + "\n"
    ff.Initialize()
    e0 = ff.CalcEnergy()
    g0 = ff.CalcGrad()
    out.append("$ENERGY %s" % fmt(e0))
    out.append("$GRAD")
    for i in range(n):
        out.append(" ".join(fmt(g0[3 * i + k]) for k in range(3)))

    # 10 iterations and full minimization (maxIters = 200), from x0 each time.
    # The final energy is evaluated with the force field built at x0 (the one
    # that was minimized; its vdW pair list depends on the build geometry).
    for iters in (10, 200):
        m2 = Chem.Mol(mol)
        c2 = m2.GetConformer()
        for i, p in enumerate(x0):
            c2.SetAtomPosition(i, p)
        rc = FFH.UFFOptimizeMolecule(
            m2, maxIters=iters, vdwThresh=vdw_thresh, ignoreInterfragInteractions=interfrag
        )
        m3 = Chem.Mol(mol)
        c3 = m3.GetConformer()
        for i, p in enumerate(x0):
            c3.SetAtomPosition(i, p)
        ff3 = FFH.UFFGetMoleculeForceField(
            m3, vdwThresh=vdw_thresh, ignoreInterfragInteractions=interfrag
        )
        ff3.Initialize()
        rc3 = ff3.Minimize(maxIts=iters)
        # NB: CalcEnergy() without arguments reuses RDKit's cached distance
        # matrix (possibly from a rejected line-search point), so pass the
        # positions explicitly.
        flat = list(ff3.Positions())
        e = ff3.CalcEnergy(flat)
        x = pts(flat, n)
        assert rc3 == rc and x == pts(c2, n), name
        out.append("$OPT %d %d %s" % (iters, rc, fmt(e)))
        for p in x:
            out.append(" ".join(fmt(v) for v in p))
    out.append("$END")
    return "\n".join(out) + "\n"


def main():
    rd = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RDKIT_SRC")
    if not rd:
        sys.exit("usage: gen_uff_reference.py RDKIT_SOURCE_DIR")
    code = os.path.join(rd, "Code")
    mmff = os.path.join(code, "ForceField", "MMFF", "test_data")
    uffd = os.path.join(code, "GraphMol", "ForceFieldHelpers", "UFF", "test_data")

    entries = []  # (name, source, molblock, coords, vdw, interfrag)
    recs = sdf_records(os.path.join(mmff, "MMFF94_dative.sdf"))
    for k in range(0, len(recs), 4):
        entries.append(("dative_%d" % k, "MMFF94_dative.sdf", recs[k], None, 10.0, True))
    recs = sdf_records(os.path.join(mmff, "MMFF94_hypervalent.sdf"))
    for k in range(1, len(recs), 12):
        entries.append(("hypervalent_%d" % k, "MMFF94_hypervalent.sdf", recs[k], None, 10.0, True))
    for fn in ("bulk.sdf", "Issue62_dg.sdf"):
        recs = sdf_records(os.path.join(uffd, fn))
        for k, r in enumerate(recs):
            entries.append(("%s_%d" % (fn.split(".")[0], k), fn, r, None, 10.0, True))
    for fn in sorted(os.listdir(uffd)):
        if fn.endswith(".mol"):
            mb = open(os.path.join(uffd, fn)).read().replace("\r\n", "\n")
            entries.append((fn[:-4], fn, mb, None, 10.0, True))

    for name, smi, add_hs in SMILES:
        m = embed(smi, add_hs)
        if m is None:
            print("embedding failed:", name, file=sys.stderr)
            continue
        mb = Chem.MolToMolBlock(m)
        n = m.GetNumAtoms()
        entries.append(("smiles_" + name, "SMILES " + smi, mb, pts(m.GetConformer(), n), 10.0, True))

    # option variations
    m = embed("CCO.O.[Na+].[Cl-]", True)
    mb = Chem.MolToMolBlock(m)
    c = pts(m.GetConformer(), m.GetNumAtoms())
    entries.append(("interfrag_on", "SMILES CCO.O.[Na+].[Cl-]", mb, c, 10.0, False))
    entries.append(("vdw_thresh_4", "SMILES CCO.O.[Na+].[Cl-]", mb, c, 4.0, True))
    m = embed("CCCCCCCCCCCC(=O)NCCO", True)
    mb = Chem.MolToMolBlock(m)
    c = pts(m.GetConformer(), m.GetNumAtoms())
    entries.append(("vdw_thresh_2", "SMILES CCCCCCCCCCCC(=O)NCCO", mb, c, 2.0, True))

    # PDB / XYZ derived structures (bonds perceived by RDKit)
    for name, smi in (("pdb_alanine_dipeptide", "CC(=O)N[C@@H](C)C(=O)NC"), ("pdb_caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C")):
        m = embed(smi, True)
        pdb = Chem.MolToPDBBlock(m)
        mp = Chem.MolFromPDBBlock(pdb, removeHs=False)
        if mp is None:
            continue
        mb = Chem.MolToMolBlock(mp)
        entries.append((name, "PDB from " + smi, mb, pts(m.GetConformer(), m.GetNumAtoms()), 10.0, True))
    for name, smi, chg in (("xyz_toluene", "Cc1ccccc1", 0), ("xyz_acetate", "CC(=O)[O-]", -1), ("xyz_pyridine_n_oxide", "[O-][n+]1ccccc1", 0)):
        m = embed(smi, True)
        xyz = Chem.MolToXYZBlock(m)
        mx = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineBonds(mx, charge=chg)
        mb = Chem.MolToMolBlock(mx)
        entries.append((name, "XYZ from " + smi, mb, pts(m.GetConformer(), m.GetNumAtoms()), 10.0, True))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    nrec = 0
    with open(OUT, "w", newline="\n") as fh:
        fh.write("# UFF reference data generated by tools/gen_uff_reference.py with RDKit %s\n" % rdBase.rdkitVersion)
        for e in entries:
            r = record(*e)
            if r is None:
                print("skipped:", e[0], file=sys.stderr)
                continue
            fh.write(r)
            nrec += 1
    print("wrote %d records to %s" % (nrec, OUT))


if __name__ == "__main__":
    main()
