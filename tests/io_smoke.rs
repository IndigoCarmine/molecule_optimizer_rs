use molecule_optimizer_rs::io;

fn rdkit_dir() -> Option<std::path::PathBuf> {
    std::env::var("RDKIT_SRC").ok().map(std::path::PathBuf::from)
}

#[test]
fn parse_mmff_validation_sdf() {
    let Some(d) = rdkit_dir() else { return };
    for f in ["MMFF94_dative.sdf", "MMFF94_hypervalent.sdf"] {
        let p = d.join("Code/ForceField/MMFF/test_data").join(f);
        let text = std::fs::read_to_string(&p).unwrap();
        let res = io::molfile::read_sdf_lenient(&text);
        let bad: Vec<String> = res.iter().filter_map(|r| r.as_ref().err().map(|e| e.to_string())).collect();
        let hidden = res.iter().filter_map(|r| r.as_ref().ok()).filter(|m| m.has_hidden_hydrogens()).count();
        println!("{f}: {} records, {} errors, {} with hidden H", res.len(), bad.len(), hidden);
        for b in bad.iter().take(5) { println!("  {b}"); }
        assert!(bad.is_empty());
    }
}

#[test]
fn show_hidden_h() {
    let Some(d) = rdkit_dir() else { return };
    let p = d.join("Code/ForceField/MMFF/test_data/MMFF94_dative.sdf");
    let text = std::fs::read_to_string(&p).unwrap();
    for m in io::read_sdf(&text).unwrap() {
        for (i, a) in m.atoms().iter().enumerate() {
            if a.total_hidden_hs() > 0 {
                let nb: Vec<String> = m.atom_bonds(i).iter().map(|&(n, b)| format!("{}{:?}", m.atom(n).symbol(), m.bond(b).order)).collect();
                println!("{} atom {} {} chg {} hs {} nbrs {:?}", m.name, i, a.symbol(), a.formal_charge, a.implicit_hs, nb);
            }
        }
    }
}
