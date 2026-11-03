"""Generates hard test cases for the RDKit legacy stereo perception port
(src/perception/legacy_stereo.rs).

Usage:
    py tools/gen_legacy_stereo_cases.py

Writes
    tests/data/stereo/cases.sdf       molecules with explicit H and ETKDG 3D
                                      coordinates (V2000)
    tests/data/stereo/cases.json      chiral tags / bond stereo after reading
                                      cases.sdf with SDMolSupplier(removeHs=False)
                                      (same format as tests/data/etkdg/stereo.json)
    tests/data/stereo/cip_ranks.json  plain legacy CIP ranks
                                      (Chirality::assignAtomCIPRanks) per molecule

Requires RDKit (tested with 2026.09.1, legacy stereo perception).
"""
import json
import os

from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "tests", "data", "stereo"))

CASES = [
    # pseudo-asymmetric centres / centres chiral only through other stereo
    ("trihydroxyglutaric_r", "OC(=O)[C@H](O)[C@H](O)[C@@H](O)C(=O)O"),
    ("trihydroxyglutaric_meso", "OC(=O)[C@H](O)C(O)[C@@H](O)C(=O)O"),
    ("pseudo_asym_cl", "C[C@H](O)C(Cl)[C@@H](C)O"),
    ("pseudo_asym_cl_2", "C[C@H](O)C(Cl)[C@H](C)O"),
    ("meso_tartaric", "OC(=O)[C@H](O)[C@@H](O)C(=O)O"),
    ("ez_pseudo_center", "C/C=C/C(O)/C=C\\C"),
    ("ee_center_not_chiral", "C/C=C/C(O)/C=C/C"),
    # ring stereo
    ("cis_13_dimethylcyclohexane", "C[C@H]1CCC[C@@H](C)C1"),
    ("trans_13_dimethylcyclohexane", "C[C@H]1CCC[C@H](C)C1"),
    ("cis_4_tbu_cyclohexanol", "CC(C)(C)[C@H]1CC[C@@H](O)CC1"),
    ("trans_14_cyclohexanediol", "O[C@H]1CC[C@H](O)CC1"),
    ("cis_14_diff_subst", "C[C@H]1CC[C@@H](CC)CC1"),
    ("spiro_55_dimethyl", "C[C@H]1CCC2(CC1)CC[C@@H](C)CC2"),
    ("spiro_33", "C1CC2(C1)CC(C)C2"),
    ("dimethyl_cyclobutane_13", "C[C@H]1C[C@@H](C)C1"),
    ("methyl_ring_fused", "C[C@H]1CC2CCCC3CCCC(C1)[C@@H]23"),
    ("inositol", "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("cubane_like_bicyclo222", "C1CC2CCC1CC2"),
    ("one_one_dimethylcyclohexane", "CC1(C)CCCCC1"),
    # nitrogen
    ("n_methylaziridine", "CN1C[C@H]1C"),
    ("quinuclidine", "C1CN2CCC1CC2"),
    ("troger_base", "Cc1ccc2c(c1)CN1Cc3cc(C)ccc3N(C2)C1"),
    ("protonated_piperidine", "C[NH+]1CC[C@H](C)CC1"),
    ("chiral_ammonium", "C[N+](CC)(CCC)c1ccccc1"),
    ("n_methylpiperidine", "CN1CCCCC1"),
    # sulfur / phosphorus
    ("chiral_sulfoxide", "C[S@](=O)c1ccccc1"),
    ("dimethyl_sulfoxide_like", "CS(=O)CC"),
    ("sulfonium", "C[S+](CC)c1ccccc1"),
    ("sulfone", "CS(=O)(=O)CC"),
    ("phosphine", "CP(CC)c1ccccc1"),
    ("phosphine_oxide", "CP(=O)(CC)c1ccccc1"),
    ("phosphate_mixed", "COP(=O)(OCC)O"),
    ("sulfinamide", "CC(C)(C)[S@](=O)N"),
    # isotopes
    ("deuterio_ethanol", "[2H][C@@H](C)O"),
    ("cd2h", "[2H]C([2H])O"),
    ("cdh2", "[2H]CC"),
    ("tritio_deuterio", "[2H][C@](C)([3H])O"),
    # ring double bonds
    ("cis_cyclooctene", "C1=C\\CCCCCC/1"),
    ("trans_cyclooctene", "C1=C/CCCCCC/1"),
    ("cis_cyclodecene", "C1=C\\CCCCCCCC/1"),
    ("cyclohexene", "C1=CCCCC1"),
    ("cycloheptene_methyl", "CC1=CCCCCC1"),
    # conjugated / multiple double bonds
    ("ee_hexadiene", "C/C=C/C=C/C"),
    ("ez_hexadiene", "C/C=C/C=C\\C"),
    ("hexatriene", "C/C=C/C=C/C=C\\C"),
    ("sorbic_acid", "C/C=C/C=C/C(=O)O"),
    ("dimethylhexadiene", "CC(C)=CC=C(C)C"),
    ("isoprene_like", "C=C(C)/C=C/C"),
    # imines / oximes / amidines
    ("e_acetaldoxime", "C/C=N/O"),
    ("acetone_oxime", "CC(C)=NO"),
    ("e_imine", "C/C=N/C"),
    # explicit stereo on the N-H: an unspecified potential stereo bond is written
    # as a crossed (V2000 stereo 3) bond, which io::read_sdf does not keep
    # (crossed bonds that remain below are on bonds that are not stereo anyway)
    ("nh_imine", "C/C=N/[H]"),
    ("ketimine_nh", "C/C(CC)=N\\[H]"),
    ("hydrazone", "C/C(=N/N)c1ccccc1"),
    ("amidine", "C/N=C(/C)N(C)C"),
    ("guanidine_like", "CN=C(N)N"),
    ("azo", "c1ccccc1/N=N/c1ccccc1"),
    # symmetric, not stereo
    ("isobutene", "C=C(C)C"),
    ("trimethylethylene", "CC=C(C)C"),
    ("three_pentanol", "CCC(O)CC"),
    ("isopropylbenzene", "CC(C)c1ccccc1"),
    ("dichloroethene_e", "Cl/C=C/Cl"),
    ("fluoro_chloro_propene", "C/C=C(/F)Cl"),
    # mixtures of stereo elements
    ("chiral_plus_db", "C[C@H](O)/C=C/C"),
    ("two_db_center", "C/C=C/[C@H](C)/C=C/CC"),
    ("amino_acid_thr", "C[C@@H](O)[C@H](N)C(=O)O"),
    ("cholesterol_core", "C[C@H](CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"),
]


def embed(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    for seed in (42, 7, 1234, 99):
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = seed
        if rdDistGeom.EmbedMolecule(m, ps) >= 0:
            break
    else:
        raise RuntimeError("embedding failed for " + smi)
    if rdForceFieldHelpers.MMFFHasAllMoleculeParams(m):
        rdForceFieldHelpers.MMFFOptimizeMolecule(m, maxIters=2000)
    return m


def stereo_record(m):
    chiral = []
    for a in m.GetAtoms():
        t = a.GetChiralTag()
        chiral.append({Chem.ChiralType.CHI_TETRAHEDRAL_CW: "CW",
                       Chem.ChiralType.CHI_TETRAHEDRAL_CCW: "CCW"}.get(t, ""))
    bonds = []
    for b in m.GetBonds():
        st = b.GetStereo()
        if st in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY):
            continue
        bonds.append({"bond": b.GetIdx(), "begin": b.GetBeginAtomIdx(), "end": b.GetEndAtomIdx(),
                      "stereo": str(st).replace("STEREO", ""), "stereo_atoms": list(b.GetStereoAtoms())})
    return {
        "chiral_tags": chiral, "bond_stereo": bonds,
        "atom_rings": [list(r) for r in m.GetRingInfo().AtomRings()],
        "hybridization": [str(a.GetHybridization()) for a in m.GetAtoms()],
        "aromatic": [a.GetIsAromatic() for a in m.GetAtoms()],
    }


def plain_cip_ranks(m):
    """Legacy assignAtomCIPRanks: with no stereo present nothing is reranked."""
    c = Chem.Mol(m)
    for a in c.GetAtoms():
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if a.HasProp("_CIPRank"):
            a.ClearProp("_CIPRank")
    for b in c.GetBonds():
        b.SetBondDir(Chem.BondDir.NONE)
        if b.GetBondType() == Chem.BondType.DOUBLE:
            b.SetStereo(Chem.BondStereo.STEREONONE)
    Chem.AssignStereochemistry(c, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    if not all(a.HasProp("_CIPRank") for a in c.GetAtoms()):
        return None
    return [a.GetIntProp("_CIPRank") for a in c.GetAtoms()]


def main():
    assert Chem.GetUseLegacyStereoPerception()
    os.makedirs(OUT, exist_ok=True)
    sdf = os.path.join(OUT, "cases.sdf")
    w = Chem.SDWriter(sdf)
    w.SetForceV3000(False)
    for name, smi in CASES:
        m = embed(smi)
        m.SetProp("_Name", name)
        m.SetProp("SMILES", smi)
        w.write(m)
    w.close()

    out, ranks = {}, {}
    for m in Chem.SDMolSupplier(sdf, removeHs=False):
        name = m.GetProp("_Name")
        out[name] = stereo_record(m)
        r = plain_cip_ranks(m)
        if r is not None:
            ranks[name] = r
    for fname, obj in (("cases.json", out), ("cip_ranks.json", ranks)):
        with open(os.path.join(OUT, fname), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, separators=(",", ":"))
    print("wrote", len(out), "cases,", len(ranks), "rank vectors to", OUT)


if __name__ == "__main__":
    main()
