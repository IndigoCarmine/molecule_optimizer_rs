"""Generates RDKit reference data for the ETKDG / SMARTS port.

Usage:
    py -I tools/gen_etkdg_reference.py [stage ...]

Stages (default: all):
    molecules   tests/data/etkdg/molecules.sdf (input molecules, explicit H,
                3D coordinates so that stereo perception sees the same stereo)
    stereo      tests/data/etkdg/stereo.json (RDKit chiral tags / bond stereo
                after reading the molblocks)
    smarts      tests/data/etkdg/torsion_matches.json (ordered SubstructMatch
                results, uniquify=False and uniquify=True, for every torsion
                pattern of the v1, v2, small-ring and macrocycle tables)
    torsions    tests/data/etkdg/exp_torsions.json (GetExperimentalTorsions)
    bounds      tests/data/etkdg/bounds.json (GetMoleculeBoundsMatrix)
    embed       tests/data/etkdg/embed_seed42.json (EmbedMolecule, ETKDGv3,
                randomSeed=42, plus onlyInitialEmbedding coordinates)
    confsearch  tests/data/etkdg/confsearch.json (EmbedMultipleConfs +
                MMFFOptimizeMoleculeConfs global minimum)

Requires RDKit (tested with 2026.09.1).
"""
import json
import os
import sys

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "tests", "data", "etkdg")
DATA = os.path.join(ROOT, "src", "distgeom", "data")

# name, SMILES
MOLECULES = [
    ("ethane", "CC"),
    ("butane", "CCCC"),
    ("hexane", "CCCCCC"),
    ("isobutane", "CC(C)C"),
    ("neopentane", "CC(C)(C)C"),
    ("ethanol", "CCO"),
    ("dimethyl_ether", "COC"),
    ("methylamine", "CN"),
    ("acetic_acid", "CC(=O)O"),
    ("methyl_acetate", "CC(=O)OC"),
    ("n_methylacetamide", "CC(=O)NC"),
    ("dmf", "CN(C)C=O"),
    ("propyne", "CC#C"),
    ("benzonitrile", "N#Cc1ccccc1"),
    ("allene", "C=C=CC"),
    ("e_2_butene", "C/C=C/C"),
    ("z_2_butene", "C/C=C\\C"),
    ("e_stilbene", "c1ccc(cc1)/C=C/c1ccccc1"),
    ("z_stilbene", "c1ccc(cc1)/C=C\\c1ccccc1"),
    ("z_oxime", "C/C(=N/O)c1ccccc1"),
    ("cyclopropane", "C1CC1"),
    ("methylcyclopropane", "CC1CC1"),
    ("cyclobutane", "C1CCC1"),
    ("oxetane", "C1COC1"),
    ("epoxide_r", "C[C@@H]1CO1"),
    ("aziridine", "C1CN1"),
    ("cyclopentane", "C1CCCC1"),
    ("cyclohexane", "C1CCCCC1"),
    ("cis_14_dimethylcyclohexane", "C[C@H]1CC[C@@H](C)CC1"),
    ("trans_14_dimethylcyclohexane", "C[C@H]1CC[C@H](C)CC1"),
    ("decalin_trans", "C1CC[C@H]2CCCC[C@@H]2C1"),
    ("norbornane", "C1CC2CCC1C2"),
    ("bicyclo111", "C1C2CC1C2"),
    ("spiro_44", "C1CCC2(C1)CCCC2"),
    ("adamantane", "C1C2CC3CC1CC(C2)C3"),
    ("benzene", "c1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("phenol", "Oc1ccccc1"),
    ("anisole", "COc1ccccc1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("biphenyl", "c1ccc(cc1)-c1ccccc1"),
    ("pyridine", "c1ccncc1"),
    ("pyrrole", "c1cc[nH]c1"),
    ("imidazole", "c1cnc[nH]1"),
    ("furan", "c1ccoc1"),
    ("thiophene", "c1ccsc1"),
    ("indole", "c1ccc2[nH]ccc2c1"),
    ("l_alanine", "C[C@@H](C(=O)O)N"),
    ("glycine_zwitterion", "[NH3+]CC(=O)[O-]"),
    ("tetramethylammonium", "C[N+](C)(C)C"),
    ("l_menthol", "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O"),
    ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("ephedrine", "C[C@@H]([C@@H](c1ccccc1)O)NC"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
    ("dimethyl_sulfoxide", "CS(C)=O"),
    ("methanesulfonamide", "CS(N)(=O)=O"),
    ("trimethyl_phosphate", "COP(=O)(OC)OC"),
    ("chlorobenzene_f", "Fc1ccc(Cl)cc1Br"),
    ("trifluoroethanol", "OCC(F)(F)F"),
    ("thioanisole", "CSc1ccccc1"),
    ("morpholine", "C1COCCN1"),
    ("piperazine_amide", "CC(=O)N1CCN(C)CC1"),
    ("cyclododecane", "C1CCCCCCCCCCC1"),
    ("macrolactone_14", "O=C1CCCCCCCCCCCCO1"),
    ("cyclic_tetrapeptide", "O=C1CNC(=O)[C@H](C)NC(=O)CNC(=O)[C@H](C)N1"),
    ("macrocycle_ether", "C1COCCOCCOCCOCCO1"),
    ("lactam_10", "O=C1CCCCCCCCN1"),
    ("sertraline", "CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21"),
    ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
    ("nicotine", "CN1CCC[C@H]1c1cccnc1"),
    ("retinal_fragment", "C/C(=C\\C=O)/C=C/C1=C(C)CCCC1(C)C"),
    ("carvone_r", "CC1=CC[C@@H](CC1=O)C(C)=C"),
    ("testosterone", "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]34C)[C@@H]1CC[C@@H]2O"),
    ("penicillin_core", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"),
    ("propofol", "CC(C)c1cccc(C(C)C)c1O"),
]

# subset used for the conformer-search acceptance test
CONFSEARCH = [
    ("butane", 50), ("hexane", 100), ("ethanol", 50), ("methyl_acetate", 50),
    ("n_methylacetamide", 50), ("cyclohexane", 50), ("l_menthol", 100),
    ("ephedrine", 100), ("aspirin", 100), ("ibuprofen", 150), ("paracetamol", 50),
    ("atenolol", 300), ("nicotine", 100), ("morpholine", 50), ("piperazine_amide", 100),
    ("cyclododecane", 300), ("macrolactone_14", 300), ("glucose", 150),
    ("propofol", 100), ("diazepam", 100),
]

BOUNDS_MOLS = ["butane", "methyl_acetate", "n_methylacetamide", "e_2_butene", "z_stilbene",
               "cyclopropane", "cyclohexane", "l_alanine", "aspirin", "norbornane",
               "macrolactone_14", "imidazole", "trifluoroethanol", "propyne", "penicillin_core",
               "glycine_zwitterion", "bicyclo111", "caffeine"]

TORSION_TABLES = {
    "v1": "torsion_prefs_v1.txt",
    "v2": "torsion_prefs_v2.txt",
    "smallrings": "torsion_prefs_smallrings.txt",
    "macrocycles": "torsion_prefs_macrocycles.txt",
}


def write_json(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    print("wrote", path, os.path.getsize(path), "bytes")


def stage_molecules():
    os.makedirs(OUT, exist_ok=True)
    w = Chem.SDWriter(os.path.join(OUT, "molecules.sdf"))
    w.SetForceV3000(False)
    for name, smi in MOLECULES:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = 1234
        if rdDistGeom.EmbedMolecule(m, ps) < 0:
            raise RuntimeError("embedding failed for " + name)
        rdForceFieldHelpers.MMFFOptimizeMolecule(m, maxIters=2000)
        m.SetProp("_Name", name)
        m.SetProp("SMILES", smi)
        w.write(m)
    w.close()
    print("wrote molecules.sdf with", len(MOLECULES), "molecules")


def load_molecules():
    """Reads the molecules exactly as the Rust side does (molblock, H kept)."""
    suppl = Chem.SDMolSupplier(os.path.join(OUT, "molecules.sdf"), removeHs=False)
    mols = []
    for m in suppl:
        if m is None:
            raise RuntimeError("could not read molecule")
        mols.append(m)
    return mols


def stage_stereo(mols):
    out = {}
    for m in mols:
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
            sa = list(b.GetStereoAtoms())
            bonds.append({"bond": b.GetIdx(), "begin": b.GetBeginAtomIdx(), "end": b.GetEndAtomIdx(),
                          "stereo": str(st).replace("STEREO", ""), "stereo_atoms": sa})
        rings = [list(r) for r in m.GetRingInfo().AtomRings()]
        out[m.GetProp("_Name")] = {
            "chiral_tags": chiral, "bond_stereo": bonds, "atom_rings": rings,
            "hybridization": [str(a.GetHybridization()) for a in m.GetAtoms()],
            "aromatic": [a.GetIsAromatic() for a in m.GetAtoms()],
        }
    write_json("stereo.json", out)


def read_table(fname):
    pats = []
    with open(os.path.join(DATA, fname), encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line[0] == "#":
                continue
            pats.append(line.split()[0])
    return pats


def stage_smarts(mols):
    out = {"tables": {}, "matches": {}}
    compiled = {}
    for tname, fname in TORSION_TABLES.items():
        pats = read_table(fname)
        out["tables"][tname] = pats
        compiled[tname] = [Chem.MolFromSmarts(p) for p in pats]
    for m in mols:
        name = m.GetProp("_Name")
        per = {}
        for tname, qs in compiled.items():
            for k, q in enumerate(qs):
                all_m = m.GetSubstructMatches(q, uniquify=False, useChirality=False, maxMatches=100000)
                if not all_m:
                    continue
                uniq = m.GetSubstructMatches(q, uniquify=True, maxMatches=100000)
                per[f"{tname}:{k}"] = {"all": [list(x) for x in all_m], "unique": [list(x) for x in uniq]}
        out["matches"][name] = per
    write_json("torsion_matches.json", out)


def stage_torsions(mols):
    out = {}
    for m in mols:
        res = {}
        for pname, ps in (("etkdgv3", rdDistGeom.ETKDGv3()), ("etkdg", rdDistGeom.ETKDG()),
                          ("sretkdgv3", rdDistGeom.srETKDGv3()), ("kdg", rdDistGeom.KDG()),
                          ("etdg", rdDistGeom.ETDG())):
            tors = rdDistGeom.GetExperimentalTorsions(m, ps)
            res[pname] = [{"atoms": list(t["atomIndices"]), "V": list(t.get("V", [])),
                           "signs": list(t.get("signs", [])), "smarts": t.get("smarts", "")}
                          for t in tors]
        out[m.GetProp("_Name")] = res
    write_json("exp_torsions.json", out)


def stage_bounds(mols):
    out = {}
    byname = {m.GetProp("_Name"): m for m in mols}
    for name in BOUNDS_MOLS:
        m = byname[name]
        ps = rdDistGeom.ETKDGv3()
        raw = rdDistGeom.GetMoleculeBoundsMatrix(m, ps, doTriangleSmoothing=False)
        smooth = rdDistGeom.GetMoleculeBoundsMatrix(m, ps, doTriangleSmoothing=True)
        # plain (non-ETKDG) parameters: no macrocycle 1-4 config
        plain = rdDistGeom.GetMoleculeBoundsMatrix(m, doTriangleSmoothing=False)
        out[name] = {"raw": raw.tolist(), "smoothed": smooth.tolist(), "plain_raw": plain.tolist()}
    write_json("bounds.json", out)


def coords(m, cid=-1):
    return [list(p) for p in m.GetConformer(cid).GetPositions()]


def stage_embed(mols):
    out = {}
    for m in mols:
        name = m.GetProp("_Name")
        rec = {}
        for pname, ctor in (("etkdgv3", rdDistGeom.ETKDGv3), ("etkdg", rdDistGeom.ETKDG),
                            ("kdg", rdDistGeom.KDG), ("etdg", rdDistGeom.ETDG)):
            mm = Chem.Mol(m)
            ps = ctor()
            ps.randomSeed = 42
            cid = rdDistGeom.EmbedMolecule(mm, ps)
            rec[pname] = coords(mm) if cid >= 0 else None
        mm = Chem.Mol(m)
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = 42
        ps.onlyInitialEmbedding = True
        cid = rdDistGeom.EmbedMolecule(mm, ps)
        rec["initial"] = coords(mm) if cid >= 0 else None
        # random-coordinate start
        mm = Chem.Mol(m)
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = 42
        ps.useRandomCoords = True
        cid = rdDistGeom.EmbedMolecule(mm, ps)
        rec["random_coords"] = coords(mm) if cid >= 0 else None
        # several conformers (per-conformer seeding scheme)
        mm = Chem.Mol(m)
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = 42
        cids = list(rdDistGeom.EmbedMultipleConfs(mm, 3, ps))
        rec["multi3"] = [coords(mm, c) for c in cids]
        out[name] = rec
    write_json("embed_seed42.json", out)


def stage_confsearch(mols):
    out = {}
    byname = {m.GetProp("_Name"): m for m in mols}
    for name, n in CONFSEARCH:
        mm = Chem.Mol(byname[name])
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = 0xF00D
        cids = list(rdDistGeom.EmbedMultipleConfs(mm, n, ps))
        res = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mm, maxIters=2000)
        best = None
        energies = []
        for cid, (conv, e) in zip(cids, res):
            energies.append(e)
            if best is None or e < best[1]:
                best = (cid, e)
        out[name] = {"num_confs": n, "num_embedded": len(cids), "seed": 0xF00D,
                     "min_energy": best[1], "min_conf_index": cids.index(best[0]),
                     "min_coords": coords(mm, best[0]), "energies": energies,
                     "molblock": Chem.MolToMolBlock(mm, confId=best[0])}
        print(name, n, best[1])
    write_json("confsearch.json", out)


def main():
    print("RDKit", rdBase.rdkitVersion)
    stages = sys.argv[1:] or ["molecules", "stereo", "smarts", "torsions", "bounds", "embed", "confsearch"]
    if "molecules" in stages:
        stage_molecules()
    mols = load_molecules()
    for s in stages:
        if s == "molecules":
            continue
        globals()["stage_" + s](mols)


if __name__ == "__main__":
    main()
