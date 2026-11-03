use super::query::{AtomExpr, AtomProp, BondExpr, Cmp};
use super::*;
use crate::core::{Atom, BondOrder};

fn mol(atoms: &[u8], bonds: &[(usize, usize, BondOrder)]) -> Molecule {
    let mut m = Molecule::new();
    for (k, &z) in atoms.iter().enumerate() {
        m.add_atom(Atom::new(z), [k as f64, (k * k) as f64 * 0.1, 0.0]);
    }
    for &(a, b, o) in bonds {
        m.add_bond(a, b, o);
    }
    crate::perception::sanitize_with(
        &mut m,
        &crate::perception::SanitizeOptions {
            aromaticity: true,
            stereo_from_3d: false,
        },
    )
    .unwrap();
    m
}

/// ethanol with explicit hydrogens: C0 C1 O2, H3..H8
fn ethanol() -> Molecule {
    use BondOrder::Single as S;
    mol(
        &[6, 6, 8, 1, 1, 1, 1, 1, 1],
        &[
            (0, 1, S),
            (1, 2, S),
            (0, 3, S),
            (0, 4, S),
            (0, 5, S),
            (1, 6, S),
            (1, 7, S),
            (2, 8, S),
        ],
    )
}

/// benzene (kekulé input) + methyl: toluene without hydrogens
fn toluene() -> Molecule {
    use BondOrder::{Double as D, Single as S};
    mol(
        &[6, 6, 6, 6, 6, 6, 6],
        &[
            (0, 1, D),
            (1, 2, S),
            (2, 3, D),
            (3, 4, S),
            (4, 5, D),
            (5, 0, S),
            (0, 6, S),
        ],
    )
}

fn p(s: &str) -> Smarts {
    Smarts::parse(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

#[test]
fn atom_and_bond_counts() {
    assert_eq!(p("C").num_atoms(), 1);
    assert_eq!(p("CC(C)(C)C").num_bonds(), 4);
    assert_eq!(p("[Cl]Br").num_atoms(), 2);
    assert_eq!(p("ClC").num_atoms(), 2);
    assert_eq!(p("c1ccccc1").num_bonds(), 6);
    assert_eq!(p("C.C").num_bonds(), 0);
    assert_eq!(p("[$(C=O),$(C#N)]N").num_atoms(), 2);
    assert_eq!(p("C%10CC%10").num_bonds(), 3);
    assert_eq!(p("C%(123)CC%(123)").num_bonds(), 3);
}

#[test]
fn ring_closure_bonds_are_added_last_in_ring_number_order() {
    // RDKit: chain bonds first, then closures sorted by ring number
    let s = p("C2CC1CC2C1");
    assert_eq!(
        s.bond_atoms(),
        vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 2), (4, 0)]
    );
    // bond spec on the closing digit is used when the opening one is bare
    let s = p("C1CC=1");
    assert_eq!(
        s.mol.bonds[2].expr,
        BondExpr::Order(Some(BondOrder::Double))
    );
    // first explicit spec wins
    let s = p("C-1CC=1");
    assert_eq!(
        s.mol.bonds[2].expr,
        BondExpr::Order(Some(BondOrder::Single))
    );
    let s = p("[C:1]1@[C:2]@;-[C:3]@[C:4]1");
    assert_eq!(s.mol.bonds[3].expr, BondExpr::SingleOrAromatic);
    assert_eq!(s.bond_atoms()[3], (3, 0));
}

#[test]
fn map_numbers() {
    let s = p("[O:1]=[C:2]([N])!@;-[O:3]~[C:4]");
    assert_eq!(
        s.atom_map_numbers(),
        vec![Some(1), Some(2), None, Some(3), Some(4)]
    );
    assert_eq!(s.atom_with_map_number(3), Some(3));
    assert_eq!(s.atom_with_map_number(7), None);
    let s = p("[H:4][C:1]");
    assert_eq!(s.mol.atoms[0].expr, AtomExpr::atomic_num(1));
    assert_eq!(s.atom_with_map_number(4), Some(0));
    let s = p("[$(C(F)(F)F):2]");
    assert_eq!(s.atom_map_numbers(), vec![Some(2)]);
}

#[test]
fn primitive_semantics() {
    let a = |s: &str| p(s).mol.atoms[0].expr.clone();
    assert_eq!(a("[C]"), AtomExpr::atom_type(6, false));
    assert_eq!(a("[c]"), AtomExpr::atom_type(6, true));
    assert_eq!(a("[se]"), AtomExpr::atom_type(34, true));
    assert_eq!(a("[Se]"), AtomExpr::atomic_num(34));
    assert_eq!(a("[Na]"), AtomExpr::atomic_num(11));
    assert_eq!(a("[#7]"), AtomExpr::atomic_num(7));
    assert_eq!(a("[R]"), AtomExpr::prim(AtomProp::NumRings, Cmp::NonZero));
    assert_eq!(a("[R2]"), AtomExpr::prim(AtomProp::NumRings, Cmp::Eq(2)));
    assert_eq!(a("[r]"), AtomExpr::prim(AtomProp::InRing, Cmp::Eq(1)));
    assert_eq!(a("[r5]"), AtomExpr::prim(AtomProp::MinRingSize, Cmp::Eq(5)));
    assert_eq!(
        a("[r{9-}]"),
        AtomExpr::prim(AtomProp::MinRingSize, Cmp::AtLeast(9))
    );
    assert_eq!(
        a("[r{-8}]"),
        AtomExpr::prim(AtomProp::MinRingSize, Cmp::AtMost(8))
    );
    assert_eq!(
        a("[r{5-8}]"),
        AtomExpr::prim(AtomProp::MinRingSize, Cmp::Range(5, 8))
    );
    assert_eq!(a("[x]"), AtomExpr::prim(AtomProp::HasRingBond, Cmp::Eq(1)));
    assert_eq!(
        a("[x3]"),
        AtomExpr::prim(AtomProp::RingBondCount, Cmp::Eq(3))
    );
    assert_eq!(
        a("[D]"),
        AtomExpr::prim(AtomProp::ExplicitDegree, Cmp::Eq(1))
    );
    assert_eq!(a("[X4]"), AtomExpr::prim(AtomProp::TotalDegree, Cmp::Eq(4)));
    assert_eq!(a("[H]"), AtomExpr::atomic_num(1));
    assert_eq!(a("[H2]"), AtomExpr::prim(AtomProp::HCount, Cmp::Eq(2)));
    assert_eq!(a("[+]"), AtomExpr::prim(AtomProp::FormalCharge, Cmp::Eq(1)));
    assert_eq!(
        a("[--]"),
        AtomExpr::prim(AtomProp::FormalCharge, Cmp::Eq(-2))
    );
    assert_eq!(
        a("[-3]"),
        AtomExpr::prim(AtomProp::FormalCharge, Cmp::Eq(-3))
    );
    assert_eq!(
        a("[C@@H]"),
        AtomExpr::and(
            AtomExpr::and(AtomExpr::atom_type(6, false), AtomExpr::True),
            AtomExpr::prim(AtomProp::HCount, Cmp::Eq(1))
        )
    );
    // precedence: implicit AND > & > , > ;
    assert_eq!(
        a("[N,CH1;X3]"),
        AtomExpr::and(
            AtomExpr::or(
                AtomExpr::atom_type(7, false),
                AtomExpr::and(
                    AtomExpr::atom_type(6, false),
                    AtomExpr::prim(AtomProp::HCount, Cmp::Eq(1))
                )
            ),
            AtomExpr::prim(AtomProp::TotalDegree, Cmp::Eq(3))
        )
    );
    assert_eq!(a("[!#1]"), AtomExpr::negate(AtomExpr::atomic_num(1)));
}

#[test]
fn parse_errors() {
    for s in [
        "", "C(", "C)", "C1CC", "[C", "C=", "[C;]", "[^7]", "C11", "Q",
    ] {
        assert!(Smarts::parse(s).is_err(), "{s:?} should fail");
    }
}

#[test]
fn ethanol_matches() {
    let m = ethanol();
    assert_eq!(
        p("[CH3][CH2][OH]").find_matches(&m, false),
        vec![vec![0, 1, 2]]
    );
    assert_eq!(p("[CX4]").find_matches(&m, false), vec![vec![0], vec![1]]);
    assert_eq!(p("[D4]").find_matches(&m, false), vec![vec![0], vec![1]]);
    assert_eq!(p("[h]").find_matches(&m, false).len(), 0);
    assert_eq!(p("[#1]").find_matches(&m, false).len(), 6);
    assert_eq!(
        p("[H][C][O]").find_matches(&m, false),
        vec![vec![6, 1, 2], vec![7, 1, 2]]
    );
    // VF2-plus order: neighbours of the image of the first mapped neighbour
    assert_eq!(
        p("C~*").find_matches(&m, false),
        vec![
            vec![0, 1],
            vec![0, 3],
            vec![0, 4],
            vec![0, 5],
            vec![1, 0],
            vec![1, 2],
            vec![1, 6],
            vec![1, 7]
        ]
    );
    assert_eq!(
        p("CC").find_matches(&m, false),
        vec![vec![0, 1], vec![1, 0]]
    );
    assert_eq!(p("CC").find_matches(&m, true), vec![vec![0, 1]]);
    assert!(p("[$(CO)]C").has_match(&m));
    assert_eq!(p("[$(CO)]C").find_matches(&m, false), vec![vec![1, 0]]);
    assert!(!p("C=C").has_match(&m));
    assert!(!p("[R]").has_match(&m));
    assert!(p("[O;!R;v2]").has_match(&m));
}

#[test]
fn aromatic_matches() {
    let m = toluene();
    assert_eq!(p("c").find_matches(&m, false).len(), 6);
    assert_eq!(p("C").find_matches(&m, false), vec![vec![6]]);
    assert_eq!(p("[c;r6]:c").find_matches(&m, false).len(), 12);
    assert_eq!(p("cC").find_matches(&m, false), vec![vec![0, 6]]);
    assert_eq!(p("c-C").find_matches(&m, false), vec![vec![0, 6]]);
    assert_eq!(p("c=C").find_matches(&m, false).len(), 0);
    assert_eq!(p("c1ccccc1").find_matches(&m, true).len(), 1);
    assert_eq!(p("c1ccccc1").find_matches(&m, false).len(), 12);
    assert_eq!(p("[x2]").find_matches(&m, false).len(), 6);
    assert!(!p("[x3]").has_match(&m));
    assert_eq!(p("[r{-8}]").find_matches(&m, false).len(), 7);
    assert_eq!(p("[r{9-}]").find_matches(&m, false).len(), 0);
    assert_eq!(p("[^2]").find_matches(&m, false).len(), 6);
    assert_eq!(p("[C^3]").find_matches(&m, false), vec![vec![6]]);
    assert_eq!(p("*@*").find_matches(&m, false).len(), 12);
    assert_eq!(p("*!@*").find_matches(&m, false).len(), 2);
    // ring without ring info present: computed locally
    let mut m2 = m.clone();
    m2.ring_info = None;
    assert_eq!(p("[R]").find_matches(&m2, false).len(), 6);
}
