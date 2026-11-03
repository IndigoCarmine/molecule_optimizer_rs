"""Generate the diverse benchmark set in benches/data/set/.

For every molecule (SMILES list below, grouped by category) this writes:
  initial.sdf    ETKDGv3 initial 3D structure (explicit H, fixed seed)
  mmff_opt.sdf   RDKit MMFF94 optimized structure (from the initial one)
  uff_opt.sdf    RDKit UFF optimized structure (from the initial one)
Each record carries properties: NAME, CATEGORY, SMILES, NUM_ATOMS,
E_INITIAL (energy at the initial structure), E_MIN (energy at the end of
RDKit's minimization, full precision), E_OPT (energy at the stored 4-decimal
optimized coordinates), OPT_RESULT
(0 = converged, 1 = max iterations), OPT_MAX_ITERS.
Molecules a force field cannot handle are omitted from that force field's
file (listed in manifest.tsv).

Usage: py tools/gen_bench_set.py [outdir]
Requires RDKit (tested with 2026.09.1).
"""
import os
import sys

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

MAX_ITERS = 10000
SEED = 0xBEEF

MOLECULES = [
    # --- simple aromatics ---
    ("benzene", "aromatic", "c1ccccc1"),
    ("toluene", "aromatic", "Cc1ccccc1"),
    ("p-xylene", "aromatic", "Cc1ccc(C)cc1"),
    ("phenol", "aromatic", "Oc1ccccc1"),
    ("aniline", "aromatic", "Nc1ccccc1"),
    ("nitrobenzene", "aromatic", "O=[N+]([O-])c1ccccc1"),
    ("benzoic_acid", "aromatic", "OC(=O)c1ccccc1"),
    ("benzonitrile", "aromatic", "N#Cc1ccccc1"),
    ("hexafluorobenzene", "aromatic", "Fc1c(F)c(F)c(F)c(F)c1F"),
    ("chlorobenzene", "aromatic", "Clc1ccccc1"),
    ("bromo_iodo_benzene", "aromatic", "Brc1ccc(I)cc1"),
    ("styrene", "aromatic", "C=Cc1ccccc1"),
    # --- polycyclic aromatics ---
    ("naphthalene", "polycyclic_aromatic", "c1ccc2ccccc2c1"),
    ("anthracene", "polycyclic_aromatic", "c1ccc2cc3ccccc3cc2c1"),
    ("phenanthrene", "polycyclic_aromatic", "c1ccc2c(c1)ccc1ccccc12"),
    ("pyrene", "polycyclic_aromatic", "c1cc2ccc3cccc4ccc(c1)c2c34"),
    ("coronene", "polycyclic_aromatic", "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61"),
    ("azulene", "polycyclic_aromatic", "c1ccc2cccc2cc1"),
    ("fluorene", "polycyclic_aromatic", "c1ccc2c(c1)Cc1ccccc12"),
    # --- biaryls / torsion-rich aromatics ---
    ("biphenyl", "biaryl", "c1ccc(cc1)-c1ccccc1"),
    ("2,2'-dimethylbiphenyl", "biaryl", "Cc1ccccc1-c1ccccc1C"),
    ("terphenyl", "biaryl", "c1ccc(cc1)-c1ccc(cc1)-c1ccccc1"),
    ("stilbene_E", "biaryl", "C(=C/c1ccccc1)\\c1ccccc1"),
    ("diphenyl_ether", "biaryl", "c1ccc(Oc2ccccc2)cc1"),
    ("benzophenone", "biaryl", "O=C(c1ccccc1)c1ccccc1"),
    ("2-phenylpyridine", "biaryl", "c1ccc(-c2ccccn2)cc1"),
    # --- heteroaromatics ---
    ("pyridine", "heteroaromatic", "c1ccncc1"),
    ("pyrrole", "heteroaromatic", "c1cc[nH]c1"),
    ("furan", "heteroaromatic", "c1ccoc1"),
    ("thiophene", "heteroaromatic", "c1ccsc1"),
    ("imidazole", "heteroaromatic", "c1c[nH]cn1"),
    ("pyrazole", "heteroaromatic", "c1cn[nH]c1"),
    ("oxazole", "heteroaromatic", "c1cocn1"),
    ("thiazole", "heteroaromatic", "c1cscn1"),
    ("1,2,4-triazole", "heteroaromatic", "c1nc[nH]n1"),
    ("tetrazole", "heteroaromatic", "c1nnn[nH]1"),
    ("pyrimidine", "heteroaromatic", "c1cncnc1"),
    ("pyrazine", "heteroaromatic", "c1cnccn1"),
    ("1,3,5-triazine", "heteroaromatic", "c1ncncn1"),
    ("indole", "heteroaromatic", "c1ccc2[nH]ccc2c1"),
    ("benzimidazole", "heteroaromatic", "c1ccc2[nH]cnc2c1"),
    ("benzofuran", "heteroaromatic", "c1ccc2occc2c1"),
    ("benzothiophene", "heteroaromatic", "c1ccc2sccc2c1"),
    ("quinoline", "heteroaromatic", "c1ccc2ncccc2c1"),
    ("isoquinoline", "heteroaromatic", "c1ccc2cnccc2c1"),
    ("acridine", "heteroaromatic", "c1ccc2nc3ccccc3cc2c1"),
    ("carbazole", "heteroaromatic", "c1ccc2c(c1)[nH]c1ccccc12"),
    ("purine", "heteroaromatic", "c1ncc2[nH]cnc2n1"),
    ("2-pyridone", "heteroaromatic", "O=c1cccc[nH]1"),
    ("pyridine_N-oxide", "heteroaromatic", "[O-][n+]1ccccc1"),
    ("pyridinium", "heteroaromatic", "c1cc[nH+]cc1"),
    # --- nucleobases ---
    ("adenine", "nucleobase", "Nc1ncnc2[nH]cnc12"),
    ("guanine", "nucleobase", "Nc1nc2[nH]cnc2c(=O)[nH]1"),
    ("cytosine", "nucleobase", "Nc1cc[nH]c(=O)n1"),
    ("thymine", "nucleobase", "Cc1c[nH]c(=O)[nH]c1=O"),
    ("uracil", "nucleobase", "O=c1cc[nH]c(=O)[nH]1"),
    # --- drugs / drug-like ---
    ("aspirin", "drug", "CC(=O)Oc1ccccc1C(=O)O"),
    ("paracetamol", "drug", "CC(=O)Nc1ccc(O)cc1"),
    ("ibuprofen", "drug", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"),
    ("caffeine", "drug", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("nicotine", "drug", "CN1CCC[C@H]1c1cccnc1"),
    ("diazepam", "drug", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
    ("naproxen", "drug", "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1"),
    ("sildenafil", "drug", "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12"),
    ("imatinib", "drug", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
    ("atorvastatin", "drug", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),
    ("celecoxib", "drug", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"),
    ("lidocaine", "drug", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("propranolol", "drug", "CC(C)NC[C@H](O)COc1cccc2ccccc12"),
    ("tamoxifen", "drug", "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"),
    ("morphine", "drug", "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"),
    ("penicillin_G", "drug", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"),
    ("warfarin", "drug", "CC(=O)C[C@@H](c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    # --- charged / zwitterions ---
    ("glycine_zwitterion", "charged", "[NH3+]CC(=O)[O-]"),
    ("acetate", "charged", "CC(=O)[O-]"),
    ("tetramethylammonium", "charged", "C[N+](C)(C)C"),
    ("guanidinium", "charged", "NC(N)=[NH2+]"),
    ("benzoate", "charged", "[O-]C(=O)c1ccccc1"),
    # --- hypervalent / halogen / sulfur / phosphorus ---
    ("dimethyl_sulfoxide", "hypervalent", "CS(C)=O"),
    ("sulfolane", "hypervalent", "O=S1(=O)CCCC1"),
    ("benzenesulfonamide", "hypervalent", "NS(=O)(=O)c1ccccc1"),
    ("trimethyl_phosphate", "hypervalent", "COP(=O)(OC)OC"),
    ("triphenylphosphine", "hypervalent", "c1ccc(cc1)P(c1ccccc1)c1ccccc1"),
    ("triphenylphosphine_oxide", "hypervalent", "O=P(c1ccccc1)(c1ccccc1)c1ccccc1"),
    ("trifluoromethylbenzene", "halogen", "FC(F)(F)c1ccccc1"),
    ("diphenyl_disulfide", "hypervalent", "c1ccc(SSc2ccccc2)cc1"),
    # --- aliphatic / rings / flexible ---
    ("n-decane", "aliphatic", "CCCCCCCCCC"),
    ("cyclohexane", "aliphatic", "C1CCCCC1"),
    ("decalin_trans", "aliphatic", "C1CC[C@H]2CCCC[C@@H]2C1"),
    ("adamantane", "aliphatic", "C1C2CC3CC1CC(C2)C3"),
    ("cubane", "aliphatic", "C12C3C4C1C1C2C3C41"),
    ("cyclopropylbenzene", "aliphatic", "C1CC1c1ccccc1"),
    ("cyclooctane", "aliphatic", "C1CCCCCCC1"),
    ("cholesterol", "aliphatic", "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"),
    ("glucose", "sugar", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("sucrose", "sugar", "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O"),
    ("18-crown-6", "macrocycle", "C1COCCOCCOCCOCCOCCO1"),
    ("cyclododecane", "macrocycle", "C1CCCCCCCCCCC1"),
    ("[2.2]paracyclophane", "macrocycle", "C1Cc2ccc(cc2)CCc2ccc1cc2"),
    ("hexabenzocoronene_fragment", "polycyclic_aromatic", "c1ccc2c(c1)c1ccccc1c1ccccc21"),
    ("helicene5", "polycyclic_aromatic", "c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12"),
    ("calixarene4", "macrocycle", "Oc1c2cccc1Cc1cccc(c1O)Cc1cccc(c1O)Cc1cccc(c1O)C2"),
    ("fullerene_C60", "polycyclic_aromatic", "c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41"),
    # --- peptides ---
    ("ala_dipeptide", "peptide", "CC(=O)N[C@@H](C)C(=O)NC"),
    ("trp_cage_fragment", "peptide", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(=O)O)C(=O)O"),
    ("gly5", "peptide", "NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)O"),
    # --- dyes / extended conjugation ---
    ("azobenzene_E", "conjugated", "c1ccc(/N=N/c2ccccc2)cc1"),
    ("beta_carotene_fragment", "conjugated", "CC(C)=CC=CC(C)=CC=CC=C(C)C=CC=C(C)C"),
    ("porphine", "conjugated", "C1=CC2=NC1=CC1=CC=C(N1)C=C1C=CC(=N1)C=C1C=CC(=C2)N1"),
    ("BODIPY_core", "conjugated", "F[B-]1(F)n2cccc2C=C2C=CC=[N+]12"),
]


def embed(mol):
    """ETKDGv3, then ETKDGv3 with random coordinates, then ETDG with random
    coordinates (needed for strained cages such as C60 or cyclophanes)."""
    for name, make, rand in (('ETKDGv3', AllChem.ETKDGv3, False), ('ETKDGv3_random', AllChem.ETKDGv3, True), ('ETDG_random', AllChem.ETDG, True)):
        p = make()
        p.randomSeed = SEED
        p.useRandomCoords = rand
        if rand:
            p.ignoreSmoothingFailures = True
        if AllChem.EmbedMolecule(mol, p) == 0:
            return name
    return None


def roundtrip(mol):
    return Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=False)


def set_props(mol, name, props):
    mol.SetProp("_Name", name)
    for k, v in props.items():
        mol.SetProp(k, v)
    mol.SetIntProp("NUM_ATOMS", mol.GetNumAtoms())


def energy(ff_kind, mol):
    if ff_kind == "MMFF":
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            return None, None
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    else:
        if not AllChem.UFFHasAllMoleculeParams(mol):
            return None, None
        ff = AllChem.UFFGetMoleculeForceField(mol)
    if ff is None:
        return None, None
    return ff, ff.CalcEnergy(ff.Positions())


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("benches", "data", "set")
    os.makedirs(outdir, exist_ok=True)
    w_init = Chem.SDWriter(os.path.join(outdir, "initial.sdf"))
    w_opt = {k: Chem.SDWriter(os.path.join(outdir, f"{k.lower()}_opt.sdf")) for k in ("MMFF", "UFF")}
    manifest = ["name\tcategory\tatoms\tmmff\tuff\tsmiles"]
    for name, cat, smi in MOLECULES:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print("bad smiles", name, file=sys.stderr)
            continue
        mh = Chem.AddHs(m)
        method = embed(mh)
        if method is None:
            print("embed failed", name, file=sys.stderr)
            manifest.append(f"{name}\t{cat}\t{mh.GetNumAtoms()}\tembed_failed\tembed_failed\t{smi}")
            continue
        # Round-trip through a molblock first, so every energy below is computed
        # from exactly the (4-decimal) coordinates stored in the SD files.
        mh = roundtrip(mh)
        props = {"EMBED_METHOD": method, "NAME": name, "CATEGORY": cat, "SMILES": smi}
        set_props(mh, name, props)
        init = Chem.Mol(mh)
        status = {}
        for kind in ("MMFF", "UFF"):
            work = Chem.Mol(mh)
            ff, e0 = energy(kind, work)
            if ff is None:
                status[kind] = "no_params"
                continue
            res = ff.Minimize(maxIts=MAX_ITERS)
            pos = list(ff.Positions())
            # energy from positions (avoids RDKit's cached distance matrix)
            e_min = ff.CalcEnergy(pos)
            conf = work.GetConformer()
            for i in range(work.GetNumAtoms()):
                conf.SetAtomPosition(i, (pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]))
            opt = roundtrip(work)
            _, e_opt = energy(kind, Chem.Mol(opt))
            set_props(opt, name, props)
            opt.SetDoubleProp("E_INITIAL", e0)
            opt.SetDoubleProp("E_MIN", e_min)
            opt.SetDoubleProp("E_OPT", e_opt)
            opt.SetIntProp("OPT_RESULT", res)
            opt.SetIntProp("OPT_MAX_ITERS", MAX_ITERS)
            init.SetDoubleProp(f"E_INITIAL_{kind}", e0)
            w_opt[kind].write(opt)
            status[kind] = "ok" if res == 0 else "not_converged"
        w_init.write(init)
        manifest.append(f"{name}\t{cat}\t{mh.GetNumAtoms()}\t{status.get('MMFF')}\t{status.get('UFF')}\t{smi}")
    w_init.close()
    for w in w_opt.values():
        w.close()
    with open(os.path.join(outdir, "manifest.tsv"), "w", newline="\n") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"RDKit {rdBase.rdkitVersion}: {len(manifest) - 1} molecules -> {outdir}")


if __name__ == "__main__":
    main()
