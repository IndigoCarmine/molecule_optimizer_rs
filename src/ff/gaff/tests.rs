//! Tests of the GAFF implementation against values that follow from the
//! paper (Wang et al. 2004) and the bundled parameter files.

use super::params::WILD;
use super::*;
use crate::core::{Atom, BondOrder};
use crate::distgeom::{EmbedParams, embed_molecule};
use crate::ff::{ForceField, gradient_error};
use crate::optim::NoObserver;
use crate::perception::{hydrogens::add_hydrogens, sanitize};

/// Heavy-atom graph (`(atomic number, formal charge)`, bonds with order
/// 1-3) completed with hydrogens. The coordinates are placeholders.
fn graph(atoms: &[(u8, i8)], bonds: &[(usize, usize, u8)]) -> Molecule {
    let mut m = Molecule::new();
    for (i, &(z, q)) in atoms.iter().enumerate() {
        let mut a = Atom::new(z);
        a.formal_charge = q;
        let t = i as f64;
        m.add_atom(a, [1.4 * t, (1.7 * t).sin(), (2.3 * t).cos()]);
    }
    for &(a, b, o) in bonds {
        let order = match o {
            1 => BondOrder::Single,
            2 => BondOrder::Double,
            _ => BondOrder::Triple,
        };
        m.add_bond(a, b, order);
    }
    sanitize(&mut m).unwrap();
    add_hydrogens(&mut m);
    sanitize(&mut m).unwrap();
    m
}

/// Neutral atoms.
fn neutral(z: &[u8], bonds: &[(usize, usize, u8)]) -> Molecule {
    graph(&z.iter().map(|&z| (z, 0)).collect::<Vec<_>>(), bonds)
}

/// [`neutral`] with embedded 3D coordinates.
fn mol3d(z: &[u8], bonds: &[(usize, usize, u8)]) -> Molecule {
    let mut m = neutral(z, bonds);
    m.coords = embed_molecule(&m, &EmbedParams::etkdg_v3(), &mut NoObserver).unwrap();
    m
}

fn types(m: &Molecule) -> Vec<String> {
    gaff_atom_types(m, GaffVariant::Gaff).unwrap()
}

/// Types of the first `n` (heavy) atoms.
fn heavy(m: &Molecule, n: usize) -> Vec<String> {
    types(m)[..n].to_vec()
}

/// Types of the hydrogens bonded to atom `i`.
fn h_on(m: &Molecule, i: usize) -> Vec<String> {
    let t = types(m);
    m.neighbors(i).filter(|&j| m.atom(j).atomic_num == 1).map(|j| t[j].clone()).collect()
}

/// A ring of `z` with alternating double bonds starting at the bond 0-1
/// (`doubles` lists the first atom of every double bond).
fn ring(z: &[u8], doubles: &[usize]) -> Vec<(usize, usize, u8)> {
    let n = z.len();
    (0..n).map(|i| (i, (i + 1) % n, if doubles.contains(&i) { 2 } else { 1 })).collect()
}

#[test]
fn parameter_files_parse() {
    for (v, n_types) in [(GaffVariant::Gaff, 83), (GaffVariant::Gaff211, 98), (GaffVariant::Gaff2, 100)] {
        let p = GaffParams::get(v);
        assert_eq!(p.atom_types().len(), n_types, "{v:?}");
        assert!(p.has_type("c3") && p.has_type("sy") && !p.has_type("X"));
        assert!(p.vdw_param("c3").is_some() && p.vdw_param("ho").is_some());
        assert!(p.angle_scale > 2.0 && p.angle_scale < 4.0, "{}", p.angle_scale);
    }
    let p = GaffParams::get(GaffVariant::Gaff);
    assert_eq!(p.bond_param("c3", "c3"), Some(BondParam { k: 300.9, r0: 1.5375 }));
    assert_eq!(p.bond_param("hc", "c3"), p.bond_param("c3", "hc"));
    assert_eq!(p.angle_param("c3", "c3", "hc"), Some(AngleParam { k: 46.3, theta0: 109.80 }));
    assert_eq!(p.angle_param("hc", "c3", "c3"), p.angle_param("c3", "c3", "hc"));
    // Fig. 3 of the paper: X-c-c-X (4, 1.2, 180.0, 2.0)
    assert_eq!(p.torsion_param("X", "c", "c", "X").unwrap(), [TorsionTerm { k: 0.3, phase: 180.0, n: 2 }]);
    let t = p.torsion_param("c3", "c3", "c3", "c3").unwrap();
    assert_eq!(
        t,
        [TorsionTerm { k: 0.18, phase: 0.0, n: 3 }, TorsionTerm { k: 0.25, phase: 180.0, n: 2 }, TorsionTerm { k: 0.2, phase: 180.0, n: 1 }]
    );
    assert_eq!(p.vdw_param("h1"), Some(VdwParam { r_star: 1.3870, eps: 0.0157 }));
    assert_eq!(p.impropers().len(), 38);
    assert!(p.impropers().iter().all(|(k, t)| k[2] != WILD && t.n == 2 && t.phase == 180.0));
    assert!(!p.has_type("c5") && GaffParams::get(GaffVariant::Gaff2).has_type("c5"));
}

/// Table 5 of the paper: the 200 generic torsions `X-j-k-X` (central
/// types; "or" alternatives of the paired types are listed separately).
const TABLE5: &[(&str, &[&str])] = &[
    ("c", &["c", "na", "ne", "nf", "no", "p3", "pe", "pf", "px", "py", "sh", "ss", "sx", "sy"]),
    ("c2", &["n", "n4", "na", "nh", "no", "p3", "sh", "ss"]),
    ("c3", &["p2", "p3", "p4", "p5", "s4", "s6"]),
    ("ca", &["n", "n3", "n4", "na", "ne", "nf", "nh", "no", "p3", "pe", "pf", "px", "py", "sh", "ss", "sx", "sy"]),
    ("ce", &["ne", "pe", "px", "py", "sx", "sy"]),
    ("cf", &["nf", "pf", "px", "py", "sx", "sy"]),
    ("n", &["n", "n2", "n3", "n4", "na", "nh", "no", "oh", "os", "p2", "p3", "p4", "p5", "s4", "s6", "sh", "ss"]),
    ("n2", &["n3", "n4", "na", "nh", "no", "oh", "os", "p3", "sh", "ss"]),
    ("n3", &["n3", "n4", "na", "nh", "no", "oh", "os", "p2", "p3", "p4", "p5", "s4", "s6", "sh", "ss"]),
    ("n4", &["n4", "na", "nh", "no", "oh", "os", "p2", "p3", "p4", "p5", "s4", "s6", "sh", "ss"]),
    ("na", &["na", "nh", "no", "oh", "os", "p2", "p3", "p4", "p5", "s4", "s6", "sh", "ss"]),
    ("ne", &["ne", "pe", "px", "py", "sx", "sy"]),
    ("nf", &["nf", "pf", "px", "py", "sx", "sy"]),
    ("nh", &["nh", "no", "oh", "os", "p2", "p3", "p4", "p5", "s4", "s6", "sh", "ss"]),
    ("no", &["no", "oh", "os", "p2", "p3", "p4", "p5", "s4", "s6", "sh", "ss"]),
    ("oh", &["oh", "os", "p2", "p3", "p4", "p5", "s4", "s6", "sh", "ss"]),
    ("os", &["os", "p2", "p3", "p4", "p5", "s4", "s6", "sh", "ss"]),
    ("p2", &["p3", "sh", "ss"]),
    ("p3", &["p3", "p4", "p5", "s4", "s6", "sh", "ss"]),
    ("p4", &["sh", "ss"]),
    ("p5", &["sh", "ss"]),
    ("pe", &["pe", "px", "py", "sx", "sy"]),
    ("pf", &["pf", "px", "py", "sx", "sy"]),
    ("px", &["px", "py", "sx", "sy"]),
    ("py", &["py", "sx", "sy"]),
    ("s4", &["sh", "ss"]),
    ("s6", &["sh", "ss"]),
    ("sh", &["sh", "ss"]),
    ("ss", &["ss"]),
    ("sx", &["sx", "sy"]),
    ("sy", &["sy"]),
];

#[test]
fn table5_generic_torsions_are_in_the_parameter_file() {
    let p = GaffParams::get(GaffVariant::Gaff);
    let mut missing = Vec::new();
    let mut count = 0;
    for &(j, ks) in TABLE5 {
        for &k in ks {
            count += 1;
            if p.torsion_param("X", j, k, "X").is_none() {
                missing.push(format!("X-{j}-{k}-X"));
            }
        }
    }
    assert!(count >= 200, "{count}");
    // GAFF 1.81 has every generic torsion of Table 5 except these three
    assert_eq!(missing, ["X-ca-n3-X", "X-ne-py-X", "X-nf-py-X"]);
}

#[test]
fn basic_types_of_fig1a() {
    // 1: 1-fluoropropane
    let m = neutral(&[6, 6, 6, 9], &[(0, 1, 1), (1, 2, 1), (2, 3, 1)]);
    assert_eq!(heavy(&m, 4), ["c3", "c3", "c3", "f"]);
    assert_eq!(h_on(&m, 0), ["hc"; 3]);
    assert_eq!(h_on(&m, 2), ["h1"; 2]);
    // 2: Cl2C=CBr2
    let m = neutral(&[17, 6, 17, 6, 35, 35], &[(0, 1, 1), (1, 2, 1), (1, 3, 2), (3, 4, 1), (3, 5, 1)]);
    assert_eq!(heavy(&m, 6), ["cl", "c2", "cl", "c2", "br", "br"]);
    // 3: 4-methoxyphenol
    let mut b = ring(&[6; 6], &[0, 2, 4]);
    b.extend([(0, 6, 1), (6, 7, 1), (3, 8, 1)]);
    let m = neutral(&[6, 6, 6, 6, 6, 6, 8, 6, 8], &b);
    assert_eq!(heavy(&m, 9), ["ca", "ca", "ca", "ca", "ca", "ca", "os", "c3", "oh"]);
    assert_eq!(h_on(&m, 8), ["ho"]);
    assert_eq!(h_on(&m, 1), ["ha"]);
    assert_eq!(h_on(&m, 7), ["h1"; 3]);
    // 4: iodoacetaldehyde: the aldehyde hydrogen is ha
    let m = neutral(&[53, 6, 6, 8], &[(0, 1, 1), (1, 2, 1), (2, 3, 2)]);
    assert_eq!(heavy(&m, 4), ["i", "c3", "c", "o"]);
    assert_eq!(h_on(&m, 2), ["ha"]);
    // 5/6: thiol and thio acid
    let m = neutral(&[6, 6, 8, 6, 16], &[(0, 1, 1), (1, 2, 2), (1, 3, 1), (3, 4, 1)]);
    assert_eq!(heavy(&m, 5), ["c3", "c", "o", "c3", "sh"]);
    assert_eq!(h_on(&m, 4), ["hs"]);
    // 7/8: thioether next to an alkyne, disulfide
    let m = neutral(&[6, 16, 6, 6, 6, 16, 16, 6], &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 3), (0, 5, 1), (5, 6, 1), (6, 7, 1)]);
    assert_eq!(heavy(&m, 8), ["c3", "ss", "c3", "c1", "c1", "ss", "ss", "c3"]);
    // 9/10: sulfoxide, sulfonamide
    let m = neutral(&[6, 16, 8, 6], &[(0, 1, 1), (1, 2, 2), (1, 3, 1)]);
    assert_eq!(heavy(&m, 4), ["c3", "s4", "o", "c3"]);
    let m = neutral(&[6, 16, 8, 8, 7, 6], &[(0, 1, 1), (1, 2, 2), (1, 3, 2), (1, 4, 1), (4, 5, 1)]);
    assert_eq!(heavy(&m, 6), ["c3", "s6", "o", "o", "n3", "c3"]);
    assert_eq!(h_on(&m, 4), ["hn"]);
    // 11: acetamide
    let m = neutral(&[6, 6, 8, 7], &[(0, 1, 1), (1, 2, 2), (1, 3, 1)]);
    assert_eq!(heavy(&m, 4), ["c3", "c", "o", "n"]);
    assert_eq!(h_on(&m, 3), ["hn"; 2]);
    // 12: amine, oxime
    let m = neutral(&[7, 6, 6, 7, 8], &[(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1)]);
    assert_eq!(heavy(&m, 5), ["n3", "c3", "c2", "n2", "oh"]);
    // 13: methylammonium
    let m = graph(&[(6, 0), (7, 1)], &[(0, 1, 1)]);
    assert_eq!(heavy(&m, 2), ["c3", "n4"]);
    assert_eq!(h_on(&m, 0), ["hx"; 3]);
    // 14: N-methylpyrrole
    let mut b = ring(&[7, 6, 6, 6, 6], &[1, 3]);
    b.push((0, 5, 1));
    let m = neutral(&[7, 6, 6, 6, 6, 6], &b);
    assert_eq!(heavy(&m, 6), ["na", "cc", "cd", "cd", "cc", "c3"]);
    // 15: aniline
    let mut b = ring(&[6; 6], &[0, 2, 4]);
    b.push((0, 6, 1));
    let m = neutral(&[6, 6, 6, 6, 6, 6, 7], &b);
    assert_eq!(heavy(&m, 7)[6], "nh");
    // 16: nitro group (charge separated) on a phosphaalkene
    let m = graph(&[(6, 0), (15, 0), (6, 0), (7, 1), (8, 0), (8, -1)], &[(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (3, 5, 1)]);
    assert_eq!(heavy(&m, 6), ["c3", "p2", "c2", "no", "o", "o"]);
    // 17-19: phosphine, phosphine oxide, phosphate
    let m = neutral(&[6, 15], &[(0, 1, 1)]);
    assert_eq!(heavy(&m, 2), ["c3", "p3"]);
    assert_eq!(h_on(&m, 1), ["hp"; 2]);
    // (CH3)2P(=O)H has four substituents on P: p5 (p4 has three)
    let m = neutral(&[6, 15, 8, 6], &[(0, 1, 1), (1, 2, 2), (1, 3, 1)]);
    assert_eq!(heavy(&m, 4), ["c3", "p5", "o", "c3"]);
    assert_eq!(h_on(&m, 1), ["hp"]);
    let m = neutral(&[6, 15, 8, 8, 8, 6], &[(0, 1, 1), (1, 2, 2), (1, 3, 1), (1, 4, 1), (4, 5, 1)]);
    assert_eq!(heavy(&m, 6), ["c3", "p5", "o", "oh", "os", "c3"]);
}

#[test]
fn special_types_of_fig1b() {
    // 1: butadiene
    let m = neutral(&[6; 4], &[(0, 1, 2), (1, 2, 1), (2, 3, 2)]);
    assert_eq!(heavy(&m, 4), ["c2", "ce", "ce", "c2"]);
    // 3: hexatriene: single bonds join equal types, double bonds different ones
    let m = neutral(&[6; 6], &[(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2)]);
    assert_eq!(heavy(&m, 6), ["c2", "ce", "ce", "cf", "cf", "c2"]);
    // 2: 2-aza-butadiene
    let m = neutral(&[6, 6, 7, 6], &[(0, 1, 2), (1, 2, 1), (2, 3, 2)]);
    assert_eq!(heavy(&m, 4), ["c2", "ce", "ne", "c2"]);
    // 4: vinylacetylene
    let m = neutral(&[6; 4], &[(0, 1, 3), (1, 2, 1), (2, 3, 2)]);
    assert_eq!(heavy(&m, 4), ["c1", "cg", "ce", "c2"]);
    // 5: 2-phosphabutadiene
    let m = neutral(&[6, 6, 15, 6], &[(0, 1, 2), (1, 2, 1), (2, 3, 2)]);
    assert_eq!(heavy(&m, 4), ["c2", "ce", "pe", "c2"]);
    // 6: furan
    let m = neutral(&[8, 6, 6, 6, 6], &ring(&[8, 6, 6, 6, 6], &[1, 3]));
    assert_eq!(heavy(&m, 5), ["os", "cc", "cd", "cd", "cc"]);
    // 7: thiazole
    let m = neutral(&[16, 6, 7, 6, 6], &ring(&[16, 6, 7, 6, 6], &[1, 3]));
    assert_eq!(heavy(&m, 5), ["ss", "cc", "nd", "cd", "cc"]);
    // 10: biphenyl
    let mut b = ring(&[6; 6], &[0, 2, 4]);
    b.extend(ring(&[6; 6], &[0, 2, 4]).into_iter().map(|(i, j, o)| (i + 6, j + 6, o)));
    b.push((0, 6, 1));
    let m = neutral(&[6; 12], &b);
    let t = heavy(&m, 12);
    assert_eq!((t[0].as_str(), t[6].as_str()), ("cp", "cp"));
    assert!(t.iter().enumerate().all(|(i, t)| i == 0 || i == 6 || t == "ca"));
    // 11: 2-phenylpyridine
    let mut z = vec![6u8; 12];
    z[7] = 7;
    let m = neutral(&z, &b);
    let t = heavy(&m, 12);
    assert_eq!((t[0].as_str(), t[6].as_str(), t[7].as_str()), ("cp", "cp", "nb"));
    // 14: naphthalene
    let m = neutral(
        &[6; 10],
        &[(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (5, 0, 1), (4, 6, 1), (6, 7, 2), (7, 8, 1), (8, 9, 2), (9, 5, 1)],
    );
    assert_eq!(heavy(&m, 10), ["ca"; 10]);
    // 17: methyl vinyl sulfoxide
    let m = neutral(&[6, 16, 8, 6, 6], &[(0, 1, 1), (1, 2, 2), (1, 3, 1), (3, 4, 2)]);
    assert_eq!(heavy(&m, 5), ["c3", "sx", "o", "ce", "c2"]);
    // 18: cyclopropyl cyclobutenyl sulfone: s6 between sp3 ring carbons
    let m = neutral(
        &[6, 6, 6, 16, 8, 8, 6, 6, 6, 6],
        &[(0, 1, 1), (1, 2, 1), (2, 0, 1), (0, 3, 1), (3, 4, 2), (3, 5, 2), (3, 6, 1), (6, 7, 1), (7, 8, 2), (8, 9, 1), (9, 6, 1)],
    );
    assert_eq!(heavy(&m, 10), ["cx", "cx", "cx", "s6", "o", "o", "cy", "cv", "cv", "cy"]);
    // 19: cyclopropenyl vinyl sulfone
    let m = neutral(
        &[6, 6, 6, 16, 8, 8, 6, 6],
        &[(0, 1, 2), (1, 2, 1), (2, 0, 1), (0, 3, 1), (3, 4, 2), (3, 5, 2), (3, 6, 1), (6, 7, 2)],
    );
    assert_eq!(heavy(&m, 8), ["cu", "cu", "cx", "sy", "o", "o", "ce", "c2"]);
    // 21: vinyl phosphine oxide (four substituents on P: py)
    let m = neutral(&[6, 15, 8, 6, 6], &[(0, 1, 1), (1, 2, 2), (1, 3, 1), (3, 4, 2)]);
    assert_eq!(heavy(&m, 5), ["c3", "py", "o", "ce", "c2"]);
}

#[test]
fn hydrogen_types_and_later_types() {
    // chloroform: three electron-withdrawing groups
    let m = neutral(&[6, 17, 17, 17], &[(0, 1, 1), (0, 2, 1), (0, 3, 1)]);
    assert_eq!(h_on(&m, 0), ["h3"]);
    // pyridine: h4 next to N
    let m = neutral(&[7, 6, 6, 6, 6, 6], &ring(&[7, 6, 6, 6, 6, 6], &[0, 2, 4]));
    assert_eq!(heavy(&m, 6), ["nb", "ca", "ca", "ca", "ca", "ca"]);
    assert_eq!(h_on(&m, 1), ["h4"]);
    assert_eq!(h_on(&m, 3), ["ha"]);
    // water
    let m = neutral(&[8], &[]);
    assert_eq!(types(&m), ["ow", "hw", "hw"]);
    // types that only the GAFF2 files have
    let m = neutral(&[6, 6, 8, 7, 6], &[(0, 1, 1), (1, 2, 2), (1, 3, 1), (3, 4, 1)]);
    assert_eq!(heavy(&m, 5)[3], "n");
    assert_eq!(gaff_atom_types(&m, GaffVariant::Gaff2).unwrap()[3], "ns");
    let m = neutral(&[6; 6], &ring(&[6; 6], &[]));
    assert_eq!(heavy(&m, 6), ["c3"; 6]);
    assert_eq!(gaff_atom_types(&m, GaffVariant::Gaff2).unwrap()[..6], ["c6"; 6]);
    // guanidine carbon, oxirane oxygen
    let m = neutral(&[6, 7, 7, 7], &[(0, 1, 2), (0, 2, 1), (0, 3, 1)]);
    assert_eq!(heavy(&m, 1), ["cz"]);
    let m = neutral(&[8, 6, 6], &ring(&[8, 6, 6], &[]));
    assert_eq!(heavy(&m, 3), ["op", "cx", "cx"]);
    // silicon has no GAFF type
    let m = neutral(&[14, 6], &[(0, 1, 1)]);
    let a = gaff_assign(&m, GaffVariant::Gaff).unwrap();
    assert_eq!(a.types[0], "");
    assert!(a.untyped.contains(&0) && !a.is_complete());
    assert!(!gaff_has_all_params(&m, GaffVariant::Gaff));
}

#[test]
fn conjugated_ring_that_cannot_be_typed_warns() {
    // The paired types cannot describe a conjugated ring with an odd number
    // of double bonds between its inner atoms (the kind of failure the
    // paper shows in Fig. 1c). Heptafulvene: three ring double bonds, and
    // the seventh ring atom is inner through its exocyclic double bond.
    let mut b = ring(&[6; 7], &[1, 3, 5]);
    b.push((0, 7, 2));
    let m = neutral(&[6; 8], &b);
    let a = gaff_assign(&m, GaffVariant::Gaff).unwrap();
    assert!(a.types.iter().all(|t| !t.is_empty()));
    assert!(a.types[..7].iter().all(|t| t == "cc" || t == "cd"), "{:?}", a.types);
    assert_eq!(a.types[7], "c2");
    assert_eq!(a.warnings.len(), 1, "{:?}", a.warnings);
    assert!(a.warnings[0].contains("paired atom types"));
    let butadiene = neutral(&[6; 4], &[(0, 1, 2), (1, 2, 1), (2, 3, 2)]);
    assert!(gaff_assign(&butadiene, GaffVariant::Gaff).unwrap().warnings.is_empty());
}

#[test]
fn fast_exact_path_gives_the_same_energy() {
    use crate::fast::{FastBuilder, FastOptions};
    for (name, m) in test_set() {
        let q: Vec<f64> = (0..m.num_atoms()).map(|i| 0.1 * ((i % 5) as f64 - 2.0)).collect();
        let b = GaffBuilder::default().with_charges(q);
        let pos = m.flat_coords();
        let reference = b.build(&m, &[]).unwrap();
        let fast = FastBuilder::new(b, FastOptions::exact()).build_dyn(&m, &[]).unwrap();
        let (mut g0, mut g1) = (vec![0.0; pos.len()], vec![0.0; pos.len()]);
        let (e0, e1) = (reference.energy_grad(&pos, &mut g0), fast.energy_grad(&pos, &mut g1));
        assert!((e0 - e1).abs() < 1e-9 * (1.0 + e0.abs()), "{name}: {e0} vs {e1}");
        let dg = g0.iter().zip(&g1).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(dg < 1e-8, "{name}: gradient differs by {dg}");
    }
}

fn ethane() -> Molecule {
    mol3d(&[6, 6], &[(0, 1, 1)])
}

#[test]
fn ethane_energy_matches_hand_computation() {
    let m = ethane();
    let b = GaffBuilder::default();
    let a = b.assign(&m).unwrap();
    assert_eq!(a.types, ["c3", "c3", "hc", "hc", "hc", "hc", "hc", "hc"]);
    assert!(a.is_complete() && a.estimated_bonds.is_empty() && a.estimated_angles.is_empty());
    assert_eq!(a.warnings.len(), 1, "{:?}", a.warnings);
    assert!(a.warnings[0].contains("no partial charges"));

    let ff = b.build(&m, &[]).unwrap();
    let pos = m.flat_coords();
    let bd = ff.energy_breakdown(&pos);
    let labels: Vec<&str> = bd.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(labels, ["BondStretch", "AngleBend", "Torsion", "VdW"]);

    use crate::core::geometry::{angle, dihedral, dist};
    let c = &m.coords;
    let hs = |a: usize| m.neighbors(a).filter(|&j| m.atom(j).atomic_num == 1).collect::<Vec<_>>();
    let (h0, h1) = (hs(0), hs(1));
    // bonds: c3-c3 300.9 / 1.5375, c3-hc 330.6 / 1.0969
    let mut e_bond = 300.9 * (dist(c[0], c[1]) - 1.5375).powi(2);
    for (a, h) in h0.iter().map(|&h| (0, h)).chain(h1.iter().map(|&h| (1, h))) {
        e_bond += 330.6 * (dist(c[a], c[h]) - 1.0969).powi(2);
    }
    // angles: c3-c3-hc 46.3 / 109.80, hc-c3-hc 39.4 / 107.58
    let mut e_angle = 0.0;
    for (a, o, h) in [(0, 1, &h0), (1, 0, &h1)] {
        for (x, &hx) in h.iter().enumerate() {
            e_angle += 46.3 * (angle(c[o], c[a], c[hx]) - 109.80_f64.to_radians()).powi(2);
            for &hy in &h[x + 1..] {
                e_angle += 39.4 * (angle(c[hx], c[a], c[hy]) - 107.58_f64.to_radians()).powi(2);
            }
        }
    }
    // torsions: hc-c3-c3-hc 0.15 (1 + cos 3φ); 1-4 vdW at half strength
    let (r, eps) = (2.0 * 1.4870, 0.0157);
    let (mut e_tors, mut e_vdw) = (0.0, 0.0);
    for &x in &h0 {
        for &y in &h1 {
            e_tors += 0.15 * (1.0 + (3.0 * dihedral(c[x], c[0], c[1], c[y])).cos());
            let s = (r / dist(c[x], c[y])).powi(6);
            e_vdw += 0.5 * eps * (s * s - 2.0 * s);
        }
    }
    let get = |k: &str| bd.iter().find(|(l, _)| l == k).unwrap().1;
    assert!((get("BondStretch") - e_bond).abs() < 1e-9, "{} {e_bond}", get("BondStretch"));
    assert!((get("AngleBend") - e_angle).abs() < 1e-9, "{} {e_angle}", get("AngleBend"));
    assert!((get("Torsion") - e_tors).abs() < 1e-9, "{} {e_tors}", get("Torsion"));
    assert!((get("VdW") - e_vdw).abs() < 1e-9, "{} {e_vdw}", get("VdW"));
    assert!((ff.energy(&pos) - (e_bond + e_angle + e_tors + e_vdw)).abs() < 1e-9);
}

#[test]
fn charges_add_the_coulomb_term() {
    let m = ethane();
    let q = vec![-0.09, -0.09, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03];
    let b = GaffBuilder::default().with_charges(q.clone());
    assert!(b.assign(&m).unwrap().warnings.is_empty());
    let ff = b.build(&m, &[]).unwrap();
    let pos = m.flat_coords();
    let e_ele = ff.energy_breakdown(&pos).into_iter().find(|(k, _)| k == "Electrostatic").unwrap().1;
    // only the nine H...H 1-4 pairs are more than two bonds apart
    let mut expect = 0.0;
    for x in m.neighbors(0).filter(|&j| j != 1) {
        for y in m.neighbors(1).filter(|&j| j != 0) {
            expect += COULOMB_CONSTANT * q[x] * q[y] / 1.2 / crate::core::geometry::dist(m.coords[x], m.coords[y]);
        }
    }
    assert!((e_ele - expect).abs() < 1e-9, "{e_ele} {expect}");
    // a dielectric constant of 4 scales the term
    let mut b4 = b.clone();
    b4.dielectric_constant = 4.0;
    let e4 = b4.build(&m, &[]).unwrap().energy_breakdown(&pos).into_iter().find(|(k, _)| k == "Electrostatic").unwrap().1;
    assert!((e4 - e_ele / 4.0).abs() < 1e-12);
    assert!(GaffBuilder::default().with_charges(vec![0.0; 3]).build(&m, &[]).is_err());
}

/// Molecules that together exercise every term kind.
fn test_set() -> Vec<(&'static str, Molecule)> {
    let mut biphenyl = ring(&[6; 6], &[0, 2, 4]);
    biphenyl.extend(ring(&[6; 6], &[0, 2, 4]).into_iter().map(|(i, j, o)| (i + 6, j + 6, o)));
    biphenyl.push((0, 6, 1));
    vec![
        ("acetic acid", mol3d(&[6, 6, 8, 8], &[(0, 1, 1), (1, 2, 2), (1, 3, 1)])),
        ("N-methylacetamide", mol3d(&[6, 6, 8, 7, 6], &[(0, 1, 1), (1, 2, 2), (1, 3, 1), (3, 4, 1)])),
        ("benzene", mol3d(&[6; 6], &ring(&[6; 6], &[0, 2, 4]))),
        ("biphenyl", mol3d(&[6; 12], &biphenyl)),
        ("butadiene", mol3d(&[6; 4], &[(0, 1, 2), (1, 2, 1), (2, 3, 2)])),
        ("propyne", mol3d(&[6, 6, 6], &[(0, 1, 1), (1, 2, 3)])),
        ("dimethyl sulfone", mol3d(&[6, 16, 8, 8, 6], &[(0, 1, 1), (1, 2, 2), (1, 3, 2), (1, 4, 1)])),
    ]
}

#[test]
fn gradients_match_numeric_for_all_variants() {
    for (name, m) in test_set() {
        let n = m.num_atoms();
        let q: Vec<f64> = (0..n).map(|i| 0.1 * ((i % 5) as f64 - 2.0)).collect();
        for v in [GaffVariant::Gaff, GaffVariant::Gaff211, GaffVariant::Gaff2] {
            let b = GaffBuilder::new(v).with_charges(q.clone());
            let a = b.assign(&m).unwrap();
            assert!(a.is_complete(), "{name} {v:?}: {a:?}");
            let ff = b.build(&m, &[]).unwrap();
            let pos = m.flat_coords();
            let e = ff.energy(&pos);
            assert!(e.is_finite(), "{name} {v:?}");
            let err = gradient_error(&ff, &pos);
            assert!(err < 1e-4 * (1.0 + e.abs()), "{name} {v:?}: gradient error {err}");
            for c in ff.contribs() {
                let mut g = vec![0.0; pos.len()];
                let eg = c.energy_grad(&pos, &mut g);
                assert!((eg - c.energy(&pos)).abs() < 1e-9 * (1.0 + eg.abs()), "{name} {}", c.label());
            }
        }
    }
}

#[test]
fn impropers_keep_sp2_centres_planar() {
    let (_, m) = test_set().remove(2);
    let ff = GaffBuilder::default().build(&m, &[]).unwrap();
    let pos = m.flat_coords();
    let bd = ff.energy_breakdown(&pos);
    // X-X-ca-ha improper on each of the six CH centres
    let n_improper = ff.contribs().iter().filter(|c| c.label() == "Improper").map(|c| c.num_terms()).sum::<usize>();
    assert_eq!(n_improper, 6);
    assert!(bd.iter().any(|(k, _)| k == "Improper"));
}

#[test]
fn optimization_and_strict_mode() {
    use crate::calc::{self, OptimizeOptions};
    for (name, m) in test_set() {
        let b = GaffBuilder::default();
        let e0 = calc::energy::energy(&m, &b).unwrap().total;
        let r = calc::optimize::optimize(&m, &b, &OptimizeOptions::default()).unwrap();
        assert!(r.energy <= e0 + 1e-9, "{name}: {} > {e0}", r.energy);
        // bond lengths of the minimum stay close to the tabulated ones
        for bond in m.bonds() {
            let d = crate::core::geometry::dist(r.coords[bond.begin], r.coords[bond.end]);
            assert!(d > 0.9 && d < 1.9, "{name}: bond length {d}");
        }
    }
    let si = mol3d(&[14, 6], &[(0, 1, 1)]);
    let mut b = GaffBuilder::default();
    assert!(b.build(&si, &[]).is_ok());
    b.strict = true;
    assert!(matches!(b.build(&si, &[]), Err(Error::MissingParameters(_))));
    // implicit hydrogens are rejected
    let mut m = Molecule::new();
    m.add_atom(Atom::new(6), [0.0; 3]);
    assert!(gaff_atom_types(&m, GaffVariant::Gaff).is_err());
}

#[test]
fn missing_parameters_are_estimated() {
    // F-Br is in no parameter file: bond from eq. (3), using eq. (4)
    let mut m = Molecule::new();
    let f = m.add_atom(Atom::new(9), [0.0, 0.0, 0.0]);
    let br = m.add_atom(Atom::new(35), [1.76, 0.0, 0.3]);
    m.add_bond(f, br, BondOrder::Single);
    sanitize(&mut m).unwrap();
    let a = gaff_assign(&m, GaffVariant::Gaff).unwrap();
    assert_eq!(a.types, ["f", "br"]);
    assert_eq!(a.estimated_bonds, [[0, 1]]);
    assert!(a.is_complete());
    let ff = GaffBuilder::default().build(&m, &[]).unwrap();
    // the input bond length is the equilibrium length
    assert!(ff.energy(&m.flat_coords()).abs() < 1e-12);
    let mut pos = m.flat_coords();
    pos[3] += 0.1;
    assert!(ff.energy(&pos) > 0.0);
}

#[test]
fn assign_reports_gaff_types() {
    use crate::ff::{ForceFieldKind, assign};
    let m = ethane();
    let a = assign(&m, ForceFieldKind::Gaff2).unwrap();
    assert_eq!(a.types[..3], ["c3", "c3", "hc"]);
    assert!(a.is_complete() && a.partial_charges.is_none());
}
