//! GAFF atom typing.
//!
//! Written from the atom type definitions of Wang et al., J. Comput. Chem.
//! 25, 1157 (2004) (Table 1, the description of the special types in groups
//! I-IV and the examples of Fig. 1) and, for the types that were added
//! after the paper, from the one-line descriptions in the atom type section
//! of the bundled parameter files. A type is only assigned if the selected
//! parameter file has it; otherwise the more general type is used (`c5` →
//! `c3`, `ns` → `n`, ...).
//!
//! Points the sources leave open, and the choices made here:
//!
//! * "electron-withdrawal group" (h1-h5): a N, O, F, Cl or Br atom bonded
//!   to the carbon ([`ELECTRON_WITHDRAWING`]). Hydrogens on a carbonyl
//!   carbon are `ha` (Fig. 1a, molecule 4).
//! * "pure aromatic" (`ca`, `nb`, `pb`): a six-membered aromatic ring of C,
//!   N and P atoms in which every atom has a double bond (in the Kekulé
//!   structure) to another aromatic atom. Atoms of other aromatic rings
//!   (five-membered rings, pyridones ...) get the conjugated-ring types
//!   `cc`/`cd`, `nc`/`nd`, `pc`/`pd`, `na`.
//! * "inner" atom of a conjugated system (`cc`, `ce`, `cg`, `nc`, `ne`,
//!   `pc`, `pe`): an atom with a multiple bond that also has a single bond
//!   to an unsaturated C/N/P/S atom (not a nitro group) or to a
//!   hypervalent S/P atom.
//! * a terminal sulfur is `s` (parameter files) rather than `s2` (paper).
//! * `hb` ("H in a bridge") and `Cu` are not assigned.

use super::params::GaffParams;
use crate::core::{BondOrder, Molecule};
use crate::error::{Error, Result};
use crate::perception::{self, rings::RingInfo};

/// Elements counted as electron-withdrawing neighbours of a carbon for the
/// hydrogen types h1-h5.
pub const ELECTRON_WITHDRAWING: [u8; 5] = [7, 8, 9, 17, 35];

/// Pairs of equivalent types: bonds between atoms of the same type of a
/// pair are single bonds, bonds between the two types are double (for
/// `cp`/`cq`: aromatic) bonds.
pub const PAIRED_TYPES: [(&str, &str); 8] =
    [("cc", "cd"), ("ce", "cf"), ("cg", "ch"), ("nc", "nd"), ("ne", "nf"), ("pc", "pd"), ("pe", "pf"), ("cp", "cq")];

/// The other type of a pair of [`PAIRED_TYPES`].
pub fn paired_type(t: &str) -> Option<&'static str> {
    PAIRED_TYPES.iter().find_map(|&(a, b)| if a == t { Some(b) } else if b == t { Some(a) } else { None })
}

/// The basic type (Table 1, Nos. 1-35) that a special type refines.
pub fn basic_type(t: &str) -> Option<&'static str> {
    Some(match t {
        "cc" | "cd" | "ce" | "cf" | "cu" | "cv" | "cz" => "c2",
        "cp" | "cq" => "ca",
        "cx" | "cy" | "c5" | "c6" => "c3",
        "cg" | "ch" => "c1",
        "cs" => "c",
        "nc" | "nd" | "ne" | "nf" => "n2",
        "ns" | "nt" | "ni" | "nj" => "n",
        "n5" | "n6" | "n7" | "n8" | "n9" | "np" | "nq" => "n3",
        "nu" | "nv" | "nm" | "nn" => "nh",
        "nx" | "ny" | "nz" | "n+" | "nk" | "nl" => "n4",
        "pc" | "pd" | "pe" | "pf" => "p2",
        "px" => "p4",
        "py" => "p5",
        "sx" => "s4",
        "sy" => "s6",
        "sp" | "sq" => "ss",
        "op" | "oq" => "os",
        _ => return None,
    })
}

/// Result of [`assign_types`].
pub(crate) struct Typing {
    /// Type per atom; empty = no GAFF type.
    pub types: Vec<&'static str>,
    pub warnings: Vec<String>,
}

struct Ctx<'a> {
    m: &'a Molecule,
    /// Kekulé structure (same atom and bond indices).
    kek: Molecule,
    rings: RingInfo,
    params: &'a GaffParams,
    /// Atom / bond in a pure aromatic six-membered ring.
    pure: Vec<bool>,
    pure_bond: Vec<bool>,
    /// Atom with a multiple bond or in a pure aromatic ring.
    unsat: Vec<bool>,
}

impl Ctx<'_> {
    fn z(&self, i: usize) -> u8 {
        self.m.atom(i).atomic_num
    }
    fn deg(&self, i: usize) -> usize {
        self.m.degree(i)
    }
    fn n_h(&self, i: usize) -> usize {
        self.m.neighbors(i).filter(|&j| self.z(j) == 1).count()
    }
    fn order(&self, b: usize) -> BondOrder {
        self.kek.bond(b).order
    }
    fn n_order(&self, i: usize, o: BondOrder) -> usize {
        self.m.atom_bonds(i).iter().filter(|&&(_, b)| self.order(b) == o).count()
    }
    fn has_double(&self, i: usize) -> bool {
        self.n_order(i, BondOrder::Double) > 0
    }
    fn has_triple(&self, i: usize) -> bool {
        self.n_order(i, BondOrder::Triple) > 0
    }
    fn arom(&self, i: usize) -> bool {
        self.m.atom(i).is_aromatic
    }
    fn in_ring(&self, i: usize) -> bool {
        self.rings.is_atom_in_ring(i)
    }
    fn ring3(&self, i: usize) -> bool {
        self.rings.is_atom_in_ring_of_size(i, 3)
    }
    fn ring4(&self, i: usize) -> bool {
        self.rings.is_atom_in_ring_of_size(i, 4)
    }
    /// The first candidate that the parameter file has.
    fn pick(&self, cands: &[&'static str]) -> &'static str {
        cands.iter().copied().find(|t| self.params.has_type(t)).unwrap_or("")
    }
    /// Double bond to O or S (`Some(true)` = to S only).
    fn carbonyl(&self, i: usize) -> Option<bool> {
        let mut found = None;
        for &(j, b) in self.m.atom_bonds(i) {
            if self.order(b) == BondOrder::Double {
                match self.z(j) {
                    8 => return Some(false),
                    16 => found = Some(true),
                    _ => {}
                }
            }
        }
        found
    }
    /// An atom that a single bond conjugates with.
    fn conj_partner(&self, j: usize) -> bool {
        match self.z(j) {
            6 => self.unsat[j],
            // not a nitro group: Fig. 1a (molecule 16) and Table 5 (X-c2-no-X)
            7 => self.unsat[j] && !self.nitro(j),
            15 => self.unsat[j] || self.deg(j) >= 4,
            16 => self.unsat[j] || self.deg(j) >= 3,
            _ => false,
        }
    }
    /// Nitrogen with three neighbours, two of them terminal oxygens.
    fn nitro(&self, i: usize) -> bool {
        self.deg(i) == 3 && self.m.neighbors(i).filter(|&j| self.z(j) == 8 && self.deg(j) == 1).count() >= 2
    }
    /// Atom with a multiple bond and a single bond to a conjugating atom.
    fn inner(&self, i: usize) -> bool {
        self.m.atom_bonds(i).iter().any(|&(j, b)| self.order(b) == BondOrder::Single && self.conj_partner(j))
    }
    /// Hypervalent S / P bonded to an unsaturated C, N or P atom.
    fn hetero_conj(&self, i: usize) -> bool {
        self.m.neighbors(i).any(|j| self.deg(j) > 1 && matches!(self.z(j), 6 | 7 | 15) && self.unsat[j])
    }

    fn carbon(&self, i: usize) -> &'static str {
        let deg = self.deg(i);
        if deg >= 4 {
            return if self.ring3(i) {
                self.pick(&["cx", "c3"])
            } else if self.ring4(i) {
                self.pick(&["cy", "c3"])
            } else if self.rings.is_atom_in_ring_of_size(i, 5) {
                self.pick(&["c5", "c3"])
            } else if self.rings.is_atom_in_ring_of_size(i, 6) {
                self.pick(&["c6", "c3"])
            } else {
                "c3"
            };
        }
        if deg == 3 {
            if let Some(thio) = self.carbonyl(i) {
                return if thio { self.pick(&["cs", "c"]) } else { "c" };
            }
            if !self.arom(i) && self.m.neighbors(i).all(|j| self.z(j) == 7) && self.params.has_type("cz") {
                return "cz";
            }
            if self.pure[i] {
                // bridge atom: bonded to another pure aromatic carbon through
                // a bond that is not part of a pure aromatic ring
                let bridge = self.m.atom_bonds(i).iter().any(|&(j, b)| self.z(j) == 6 && self.pure[j] && !self.pure_bond[b]);
                return if bridge { self.pick(&["cp", "ca"]) } else { "ca" };
            }
            if self.ring3(i) {
                return self.pick(&["cu", "c2"]);
            }
            if self.ring4(i) {
                return self.pick(&["cv", "c2"]);
            }
            if self.arom(i) {
                return self.pick(&["cc", "c2"]);
            }
            if self.has_double(i) && self.inner(i) {
                return if self.in_ring(i) { self.pick(&["cc", "c2"]) } else { self.pick(&["ce", "c2"]) };
            }
            return "c2";
        }
        if deg == 2 {
            if self.has_triple(i) {
                return if self.inner(i) { self.pick(&["cg", "c1"]) } else { "c1" };
            }
            if self.n_order(i, BondOrder::Double) == 1 {
                return if self.arom(i) { self.pick(&["cc", "c2"]) } else { "c2" };
            }
        }
        "c1"
    }

    fn nitrogen(&self, i: usize) -> &'static str {
        let (deg, nh) = (self.deg(i), self.n_h(i));
        let (r3, r4) = (self.ring3(i), self.ring4(i));
        match deg {
            0 | 1 => "n1",
            2 => {
                if self.has_triple(i) || self.n_order(i, BondOrder::Double) >= 2 {
                    "n1"
                } else if self.pure[i] {
                    self.pick(&["nb", "n2"])
                } else if self.arom(i) {
                    self.pick(&["nc", "n2"])
                } else if self.has_double(i) && self.inner(i) {
                    if self.in_ring(i) { self.pick(&["nc", "n2"]) } else { self.pick(&["ne", "n2"]) }
                } else {
                    "n2"
                }
            }
            3 => {
                if self.nitro(i) {
                    return "no";
                }
                if self.pure[i] && self.has_double(i) {
                    return self.pick(&["nb", "na"]);
                }
                if self.arom(i) || self.has_double(i) {
                    return "na";
                }
                let amide = self.m.neighbors(i).any(|j| self.z(j) == 6 && self.carbonyl(j).is_some());
                if amide {
                    return match (r3, r4, nh) {
                        (true, _, _) => self.pick(&["ni", "n"]),
                        (_, true, _) => self.pick(&["nj", "n"]),
                        (_, _, 1) => self.pick(&["ns", "n"]),
                        (_, _, 2) => self.pick(&["nt", "n"]),
                        _ => "n",
                    };
                }
                if self.m.neighbors(i).any(|j| self.arom(j)) {
                    return match (r3, r4, nh) {
                        (true, _, _) => self.pick(&["nm", "nh"]),
                        (_, true, _) => self.pick(&["nn", "nh"]),
                        (_, _, 1) => self.pick(&["nu", "nh"]),
                        (_, _, 2) => self.pick(&["nv", "nh"]),
                        _ => "nh",
                    };
                }
                match (r3, r4, nh) {
                    (true, _, 1) => self.pick(&["n5", "np", "n3"]),
                    (true, _, _) => self.pick(&["np", "n3"]),
                    (_, true, 1) => self.pick(&["n6", "nq", "n3"]),
                    (_, true, _) => self.pick(&["nq", "n3"]),
                    (_, _, 1) => self.pick(&["n7", "n3"]),
                    (_, _, 2) => self.pick(&["n8", "n3"]),
                    (_, _, 3) => self.pick(&["n9", "n3"]),
                    _ => "n3",
                }
            }
            _ => match (r3, r4, nh) {
                (true, _, _) => self.pick(&["nk", "n4"]),
                (_, true, _) => self.pick(&["nl", "n4"]),
                (_, _, 1) => self.pick(&["nx", "n4"]),
                (_, _, 2) => self.pick(&["ny", "n4"]),
                (_, _, 3) => self.pick(&["nz", "n4"]),
                (_, _, 4) => self.pick(&["n+", "n4"]),
                _ => "n4",
            },
        }
    }

    fn oxygen(&self, i: usize) -> &'static str {
        let (deg, nh) = (self.deg(i), self.n_h(i));
        if deg <= 1 {
            "o"
        } else if deg == 2 && nh == 2 {
            self.pick(&["ow", "oh"])
        } else if nh > 0 {
            "oh"
        } else if self.ring3(i) {
            self.pick(&["op", "os"])
        } else if self.ring4(i) {
            self.pick(&["oq", "os"])
        } else {
            "os"
        }
    }

    fn sulfur(&self, i: usize) -> &'static str {
        match self.deg(i) {
            0 | 1 => self.pick(&["s", "s2"]),
            2 => {
                if self.n_h(i) > 0 {
                    "sh"
                } else if self.has_double(i) {
                    "s2"
                } else if self.ring3(i) {
                    self.pick(&["sp", "ss"])
                } else if self.ring4(i) {
                    self.pick(&["sq", "ss"])
                } else {
                    "ss"
                }
            }
            3 => {
                if self.hetero_conj(i) {
                    self.pick(&["sx", "s4"])
                } else {
                    "s4"
                }
            }
            4 => {
                if self.hetero_conj(i) {
                    self.pick(&["sy", "s6"])
                } else {
                    "s6"
                }
            }
            _ => "s6",
        }
    }

    fn phosphorus(&self, i: usize) -> &'static str {
        match self.deg(i) {
            0 | 1 => "p2",
            2 => {
                if self.pure[i] {
                    self.pick(&["pb", "p2"])
                } else if self.arom(i) {
                    self.pick(&["pc", "p2"])
                } else if self.has_double(i) && self.inner(i) {
                    if self.in_ring(i) { self.pick(&["pc", "p2"]) } else { self.pick(&["pe", "p2"]) }
                } else {
                    "p2"
                }
            }
            3 => {
                if self.has_double(i) || self.m.atom(i).formal_charge > 0 {
                    if self.hetero_conj(i) { self.pick(&["px", "p4"]) } else { "p4" }
                } else {
                    "p3"
                }
            }
            4 => {
                if self.hetero_conj(i) {
                    self.pick(&["py", "p5"])
                } else {
                    "p5"
                }
            }
            _ => "p5",
        }
    }

    fn hydrogen(&self, i: usize, heavy: &[&'static str]) -> &'static str {
        let Some(a) = self.m.neighbors(i).next() else { return "" };
        match self.z(a) {
            6 => {
                let ewg = self.m.neighbors(a).filter(|&j| ELECTRON_WITHDRAWING.contains(&self.z(j))).count();
                if self.deg(a) >= 4 {
                    let cation = self.m.neighbors(a).any(|j| self.z(j) == 7 && self.deg(j) == 4 && self.m.atom(j).formal_charge > 0);
                    if cation && self.params.has_type("hx") {
                        return "hx";
                    }
                    match ewg {
                        0 => "hc",
                        1 => self.pick(&["h1", "hc"]),
                        2 => self.pick(&["h2", "hc"]),
                        _ => self.pick(&["h3", "hc"]),
                    }
                } else if matches!(heavy[a], "c" | "cs") {
                    "ha"
                } else {
                    match ewg {
                        0 => "ha",
                        1 => self.pick(&["h4", "ha"]),
                        _ => self.pick(&["h5", "ha"]),
                    }
                }
            }
            7 => "hn",
            8 => {
                if heavy[a] == "ow" {
                    self.pick(&["hw", "ho"])
                } else {
                    "ho"
                }
            }
            15 => "hp",
            16 => "hs",
            _ => "",
        }
    }
}

/// Six-membered rings of C/N/P in which every atom is aromatic and has a
/// Kekulé double bond to an aromatic neighbour over an aromatic bond.
fn pure_aromatic(m: &Molecule, kek: &Molecule, rings: &RingInfo) -> (Vec<bool>, Vec<bool>) {
    let mut pure = vec![false; m.num_atoms()];
    let mut pure_bond = vec![false; m.num_bonds()];
    for (ring, bonds) in rings.atom_rings.iter().zip(&rings.bond_rings) {
        if ring.len() != 6 {
            continue;
        }
        let ok = ring.iter().all(|&i| {
            m.atom(i).is_aromatic
                && matches!(m.atom(i).atomic_num, 6 | 7 | 15)
                && m.atom_bonds(i)
                    .iter()
                    .any(|&(j, b)| kek.bond(b).order == BondOrder::Double && m.bond(b).is_aromatic && m.atom(j).is_aromatic)
        });
        if ok {
            for &i in ring {
                pure[i] = true;
            }
            for &b in bonds {
                pure_bond[b] = true;
            }
        }
    }
    (pure, pure_bond)
}

/// Distributes the atoms of `first` types over the two types of each pair:
/// `differ(bond)` tells whether the two ends of a bond between such atoms
/// get different types. Returns the atoms of bonds that cannot be
/// satisfied.
fn two_colour(m: &Molecule, member: &[bool], differ: impl Fn(usize) -> bool) -> (Vec<bool>, Vec<(usize, usize)>) {
    let n = m.num_atoms();
    let mut colour: Vec<Option<bool>> = vec![None; n];
    let mut conflicts = Vec::new();
    let mut queue = Vec::new();
    for s in 0..n {
        if !member[s] || colour[s].is_some() {
            continue;
        }
        colour[s] = Some(false);
        queue.clear();
        queue.push(s);
        let mut head = 0;
        while head < queue.len() {
            let i = queue[head];
            head += 1;
            let ci = colour[i].expect("coloured");
            for &(j, b) in m.atom_bonds(i) {
                if !member[j] {
                    continue;
                }
                let want = ci ^ differ(b);
                match colour[j] {
                    None => {
                        colour[j] = Some(want);
                        queue.push(j);
                    }
                    Some(cj) => {
                        if cj != want && i < j {
                            conflicts.push((i, j));
                        }
                    }
                }
            }
        }
    }
    (colour.into_iter().map(|c| c.unwrap_or(false)).collect(), conflicts)
}

/// Assigns GAFF atom types to `mol`, which must be in the perceived form of
/// [`crate::ff::assign::normalized`] with all hydrogens as atoms.
pub(crate) fn assign_types(mol: &Molecule, params: &GaffParams) -> Result<Typing> {
    if mol.has_hidden_hydrogens() {
        return Err(Error::InvalidArgument("GAFF needs all hydrogens as explicit atoms".into()));
    }
    let n = mol.num_atoms();
    let kek = perception::kekulized_copy(mol)?;
    let rings = match mol.ring_info() {
        Some(r) => r.clone(),
        None => RingInfo::compute(mol),
    };
    let (pure, pure_bond) = pure_aromatic(mol, &kek, &rings);
    let unsat: Vec<bool> = (0..n)
        .map(|i| pure[i] || mol.atom_bonds(i).iter().any(|&(_, b)| matches!(kek.bond(b).order, BondOrder::Double | BondOrder::Triple)))
        .collect();
    let cx = Ctx { m: mol, kek, rings, params, pure, pure_bond, unsat };

    let mut types: Vec<&'static str> = vec![""; n];
    for (i, t) in types.iter_mut().enumerate() {
        *t = match cx.z(i) {
            6 => cx.carbon(i),
            7 => cx.nitrogen(i),
            8 => cx.oxygen(i),
            9 => "f",
            15 => cx.phosphorus(i),
            16 => cx.sulfur(i),
            17 => "cl",
            35 => "br",
            53 => "i",
            _ => "",
        };
    }
    for i in 0..n {
        if cx.z(i) == 1 {
            types[i] = cx.hydrogen(i, &types);
        }
    }

    // paired types: same type across single bonds, different types across
    // multiple bonds (cp/cq: across the bonds of the aromatic rings)
    let mut warnings = Vec::new();
    let conj: Vec<bool> = types.iter().map(|t| matches!(*t, "cc" | "ce" | "cg" | "nc" | "ne" | "pc" | "pe")).collect();
    let (second, conflicts) = two_colour(mol, &conj, |b| cx.order(b) != BondOrder::Single);
    let bridge: Vec<bool> = types.iter().map(|t| *t == "cp").collect();
    let (second_bridge, bridge_conflicts) = two_colour(mol, &bridge, |b| cx.pure_bond[b]);
    for i in 0..n {
        if (conj[i] && second[i]) || (bridge[i] && second_bridge[i]) {
            let other = paired_type(types[i]).expect("paired type");
            if params.has_type(other) {
                types[i] = other;
            }
        }
    }
    for (i, j) in conflicts.into_iter().chain(bridge_conflicts) {
        warnings.push(format!(
            "atoms {i} ({}) and {j} ({}): the paired atom types cannot tell the single from the double bonds of this conjugated system",
            types[i], types[j]
        ));
    }
    for (i, t) in types.iter().enumerate() {
        if t.is_empty() {
            warnings.push(format!("atom {i} ({}) has no GAFF atom type", mol.atom(i).symbol()));
        }
    }
    Ok(Typing { types, warnings })
}
