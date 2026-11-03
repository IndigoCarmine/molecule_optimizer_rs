"""Generates reference 2D depictions for the port of RDKit's depictor
(src/depict).

Usage:
    py tools/gen_depict_reference.py

Writes
    tests/data/depict/molecules_h.sdf      test molecules with explicit H and
                                           ETKDG 3D coordinates (V2000)
    tests/data/depict/molecules_noh.sdf    the same molecules without H
                                           (3D coordinates kept)
    tests/data/depict/reference.json       rdDepictor.Compute2DCoords results
                                           for every molecule of both files and
                                           of benches/data/set/initial.sdf,
                                           read with SDMolSupplier(removeHs=False)

Variants per molecule:
    default    Compute2DCoords(mol)                     (canonOrient=True)
    noorient   Compute2DCoords(mol, canonOrient=False)
    sampled    Compute2DCoords(mol, nFlipsPerSample=3, nSample=20, sampleSeed=7)
    nopae      Compute2DCoords(mol, usePathAngleExpansion=False)
    coordmap   Compute2DCoords(mol, coordMap={...}) with two fixed atoms
    bondlen    Compute2DCoords(mol, bondLength=1.0)

Requires RDKit (tested with 2026.09.1, legacy stereo perception, CoordGen
not preferred).
"""
import json
import os

from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDistGeom
from rdkit.Geometry import Point2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "tests", "data", "depict")

CASES = [
    # chains / branching
    ("methane", "C"),
    ("ethane", "CC"),
    ("isobutane", "CC(C)C"),
    ("neopentane", "CC(C)(C)C"),
    ("icosane", "CCCCCCCCCCCCCCCCCCCC"),
    ("branched_alkane", "CC(C)CC(C)(C)CC(CC)C(C)CCC(C)(C)C"),
    ("tetraphenylmethane", "c1ccc(cc1)C(c1ccccc1)(c1ccccc1)c1ccccc1"),
    ("triphenylphosphine", "c1ccc(cc1)P(c1ccccc1)c1ccccc1"),
    ("acetonitrile", "CC#N"),
    ("diyne", "CC#CC#CC"),
    ("allene", "C=C=C"),
    ("co2", "O=C=O"),
    ("dmso", "CS(C)=O"),
    ("sulfonamide", "CS(=O)(=O)N"),
    ("sf6", "FS(F)(F)(F)(F)F"),
    ("pcl5", "ClP(Cl)(Cl)(Cl)Cl"),
    # rings
    ("cyclopropane", "C1CC1"),
    ("cyclobutane", "C1CCC1"),
    ("benzene", "c1ccccc1"),
    ("cyclododecane", "C1CCCCCCCCCCC1"),
    ("crown6", "C1COCCOCCOCCOCCOCCO1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("phenanthrene", "c1ccc2c(c1)ccc1ccccc12"),
    ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34"),
    ("coronene", "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61"),
    ("azulene", "c1ccc2cccc2cc1"),
    ("biphenyl", "c1ccc(cc1)-c1ccccc1"),
    ("ortho_terphenyl", "c1ccc(cc1)-c1ccccc1-c1ccccc1"),
    ("adamantane", "C1C2CC3CC1CC(C2)C3"),
    ("norbornane", "C1CC2CCC1C2"),
    ("bicyclo222", "C1CC2CCC1CC2"),
    ("cubane", "C12C3C4C1C5C2C3C45"),
    ("spiro45", "C1CCC2(C1)CCCCC2"),
    ("dispiro", "C1CC2(C1)CCC1(CC1)CC2"),
    ("spiro_substituted", "CC1CCC2(CC1)CCC(C)(C)CC2"),
    ("cholesterol", "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"),
    ("morphine", "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"),
    ("strychnine", "C1CN2CC3=CCO[C@H]4CC(=O)N5[C@H]6[C@H]4[C@H]3C[C@H]2[C@@]61C7=CC=CC=C75"),
    ("paclitaxel_core", "CC1=C2[C@@H](OC(C)=O)C(=O)[C@]3(C)[C@@H](O)C[C@H]4OC[C@@]4(OC(C)=O)[C@H]3[C@H](OC(=O)c3ccccc3)[C@](O)(C[C@@H]1O)C2(C)C"),
    ("macrocycle_lactone", "O=C1CCCCCCCCCCCCCCO1"),
    ("cyclic_peptide", "O=C1CNC(=O)CNC(=O)CNC(=O)CNC(=O)CN1"),
    # stereo double bonds
    ("e_butene", "C/C=C/C"),
    ("z_butene", "C/C=C\\C"),
    ("e_stilbene", "c1ccc(cc1)/C=C/c1ccccc1"),
    ("z_stilbene", "c1ccc(cc1)/C=C\\c1ccccc1"),
    ("retinal", "CC1=C(C(C)(C)CCC1)/C=C/C(C)=C/C=C/C(C)=C/C=O"),
    ("conj_diene_ez", "C/C=C/C=C\\C"),
    ("trans_cyclooctene", "C1CCC/C=C/CC1"),
    ("cis_cyclodecene", "C1CCCC/C=C\\CCC1"),
    ("oleic_acid", "CCCCCCCC/C=C\\CCCCCCCC(=O)O"),
    # tetrahedral stereo
    ("l_alanine", "C[C@@H](C(=O)O)N"),
    ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("tartaric", "OC(=O)[C@H](O)[C@@H](O)C(=O)O"),
    # drugs
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("penicillin_g", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"),
    ("atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),
    ("imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
    ("sildenafil", "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12"),
    ("vancomycin_fragment", "O=C(O)[C@@H]1NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H](N)Cc2ccc(O)c(c2)-c2cc1ccc2O"),
    # multiple fragments / ions
    ("sodium_acetate", "CC(=O)[O-].[Na+]"),
    ("salt_three", "[Na+].[Cl-].c1ccccc1"),
    ("water_dimer", "O.O"),
    ("ammonium_sulfate", "[NH4+].[NH4+].[O-]S([O-])(=O)=O"),
    ("pyridinium", "c1cc[nH+]cc1.[Cl-]"),
]


def embed(smi, name):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    ps = rdDistGeom.ETKDGv3()
    ps.randomSeed = 42
    if rdDistGeom.EmbedMolecule(mol, ps) != 0:
        ps.useRandomCoords = True
        if rdDistGeom.EmbedMolecule(mol, ps) != 0:
            raise RuntimeError(f"embedding failed: {name}")
    mol.SetProp("_Name", name)
    return mol


def coords(mol):
    conf = mol.GetConformer()
    return [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())]


def variants(mol):
    out = {}
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    out["default"] = coords(m)
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m, canonOrient=False)
    out["noorient"] = coords(m)
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m, nFlipsPerSample=3, nSample=20, sampleSeed=7)
    out["sampled"] = coords(m)
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m, usePathAngleExpansion=False)
    out["nopae"] = coords(m)
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m, bondLength=1.0)
    out["bondlen"] = coords(m)
    if mol.GetNumAtoms() >= 2:
        b = mol.GetBondWithIdx(0) if mol.GetNumBonds() else None
        if b is not None:
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            cmap = {a1: Point2D(1.0, 2.0), a2: Point2D(2.5, 2.0)}
            m = Chem.Mol(mol)
            rdDepictor.Compute2DCoords(m, coordMap=cmap)
            out["coordmap"] = {"map": {str(a1): [1.0, 2.0], str(a2): [2.5, 2.0]}, "coords": coords(m)}
    return out


def main():
    rdDepictor.SetPreferCoordGen(False)
    os.makedirs(OUT, exist_ok=True)
    h_path = os.path.join(OUT, "molecules_h.sdf")
    noh_path = os.path.join(OUT, "molecules_noh.sdf")
    wh = Chem.SDWriter(h_path)
    wn = Chem.SDWriter(noh_path)
    wh.SetKekulize(True)
    wn.SetKekulize(True)
    for name, smi in CASES:
        mh = embed(smi, name)
        wh.write(mh)
        mn = Chem.RemoveHs(mh)
        mn.SetProp("_Name", name)
        wn.write(mn)
    wh.close()
    wn.close()

    ref = {}
    sources = {
        "h": h_path,
        "noh": noh_path,
        "set": os.path.join(ROOT, "benches", "data", "set", "initial.sdf"),
    }
    for key, path in sources.items():
        ref[key] = {}
        for mol in Chem.SDMolSupplier(path, removeHs=False):
            if mol is None:
                continue
            ref[key][mol.GetProp("_Name")] = variants(mol)
    with open(os.path.join(OUT, "reference.json"), "w") as fh:
        json.dump(ref, fh, separators=(",", ":"))
    print({k: len(v) for k, v in ref.items()})


if __name__ == "__main__":
    main()
