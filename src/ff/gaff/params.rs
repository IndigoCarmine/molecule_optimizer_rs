//! GAFF parameter tables, read from the bundled `gaff*.dat` files
//! (`src/data/gaff/`, public domain; see the `README.md` there).
//!
//! The meaning of the columns follows the format notes at the end of
//! `gaff2.dat` and Fig. 3 of Wang et al. (2004):
//!
//! * bonds `E = K (r - r0)²`, angles `E = K (θ - θ0)²` (eq. 1 of the paper);
//! * torsions `multiplicity  force_constant  phase  periodicity`, with
//!   `E = (force_constant / multiplicity) (1 + cos(n φ - phase))`; a
//!   negative periodicity means that more terms of the same torsion follow;
//! * impropers `force_constant  phase  periodicity`, central atom third;
//! * van der Waals `R*`, `ε` per atom type, combined as `R*_i + R*_j` and
//!   `sqrt(ε_i ε_j)`.

use super::estimate;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Parameter set of the GAFF family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GaffVariant {
    /// GAFF 1.81 (`gaff.dat`), the force field of Wang et al. (2004) with
    /// its later updates.
    #[default]
    Gaff,
    /// GAFF 2.11 (`gaff211.dat`).
    Gaff211,
    /// GAFF 2.2.30 (`gaff2.dat`).
    Gaff2,
}

impl GaffVariant {
    pub fn name(self) -> &'static str {
        match self {
            GaffVariant::Gaff => "GAFF",
            GaffVariant::Gaff211 => "GAFF2.11",
            GaffVariant::Gaff2 => "GAFF2",
        }
    }
    fn text(self) -> &'static str {
        match self {
            GaffVariant::Gaff => include_str!("../../data/gaff/gaff.dat"),
            GaffVariant::Gaff211 => include_str!("../../data/gaff/gaff211.dat"),
            GaffVariant::Gaff2 => include_str!("../../data/gaff/gaff2.dat"),
        }
    }
}

/// Wildcard (`X`) in torsion and improper keys.
pub(crate) const WILD: u16 = u16::MAX;

/// Atom type line: name, mass and the one-line description.
#[derive(Debug, Clone)]
pub struct AtomTypeInfo {
    pub name: String,
    pub mass: f64,
    pub description: String,
}

/// `E = k (r - r0)²` (kcal/mol/Å², Å).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BondParam {
    pub k: f64,
    pub r0: f64,
}

/// `E = k (θ - θ0)²` (kcal/mol/rad², degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngleParam {
    pub k: f64,
    pub theta0: f64,
}

/// `E = k (1 + cos(n φ - phase))` (kcal/mol, degrees); `k` is already
/// divided by the multiplicity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TorsionTerm {
    pub k: f64,
    pub phase: f64,
    pub n: u32,
}

/// Van der Waals radius `R*` (Å) and well depth `ε` (kcal/mol).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VdwParam {
    pub r_star: f64,
    pub eps: f64,
}

/// One parsed parameter file.
#[derive(Debug)]
pub struct GaffParams {
    pub variant: GaffVariant,
    names: Vec<String>,
    ids: HashMap<String, u16>,
    atom_types: Vec<AtomTypeInfo>,
    is_atom_type: Vec<bool>,
    bonds: HashMap<(u16, u16), BondParam>,
    angles: HashMap<(u16, u16, u16), AngleParam>,
    torsions: HashMap<[u16; 4], Vec<TorsionTerm>>,
    impropers: Vec<([u16; 4], TorsionTerm)>,
    vdw: HashMap<u16, VdwParam>,
    /// Scale of the empirical angle force constant (see
    /// [`estimate::angle_force_constant`]), fitted to this table.
    pub(crate) angle_scale: f64,
}

fn torsion_key(k: [u16; 4]) -> [u16; 4] {
    let r = [k[3], k[2], k[1], k[0]];
    if r < k { r } else { k }
}

/// Element of an atom type name (`None` for names that are not GAFF
/// organic types, e.g. `Cu`).
fn type_element(name: &str) -> Option<u8> {
    match name {
        "cl" => Some(17),
        "br" => Some(35),
        _ => match name.as_bytes().first()? {
            b'h' => Some(1),
            b'c' => Some(6),
            b'n' => Some(7),
            b'o' => Some(8),
            b'f' => Some(9),
            b'p' => Some(15),
            b's' => Some(16),
            b'i' => Some(53),
            _ => None,
        },
    }
}

impl GaffParams {
    /// The (lazily parsed) table of `variant`.
    pub fn get(variant: GaffVariant) -> &'static GaffParams {
        static TABLES: [OnceLock<GaffParams>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
        TABLES[variant as usize].get_or_init(|| GaffParams::parse(variant, variant.text()))
    }

    fn intern(&mut self, name: &str) -> u16 {
        if name == "X" {
            return WILD;
        }
        if let Some(&i) = self.ids.get(name) {
            return i;
        }
        let i = self.names.len() as u16;
        self.names.push(name.to_string());
        self.is_atom_type.push(false);
        self.ids.insert(name.to_string(), i);
        i
    }

    fn parse(variant: GaffVariant, text: &str) -> GaffParams {
        let mut p = GaffParams {
            variant,
            names: Vec::new(),
            ids: HashMap::new(),
            atom_types: Vec::new(),
            is_atom_type: Vec::new(),
            bonds: HashMap::new(),
            angles: HashMap::new(),
            torsions: HashMap::new(),
            impropers: Vec::new(),
            vdw: HashMap::new(),
            angle_scale: 1.0,
        };
        // type names are two characters wide, separated by '-'
        fn name_at(line: &str, k: usize) -> &str {
            line.get(3 * k..(3 * k + 2).min(line.len())).unwrap_or("").trim()
        }
        fn nums(line: &str, from: usize, n: usize) -> Option<Vec<f64>> {
            let v: Vec<f64> = line.get(from..)?.split_whitespace().take(n).map_while(|t| t.parse().ok()).collect();
            (v.len() == n).then_some(v)
        }
        let mut lines = text.lines().skip(1);

        // atom types
        for l in lines.by_ref() {
            if l.trim().is_empty() {
                break;
            }
            let mut it = l.split_whitespace();
            let (Some(name), Some(mass)) = (it.next(), it.next().and_then(|m| m.parse::<f64>().ok())) else { continue };
            let _polarizability = it.next();
            let description = it.collect::<Vec<_>>().join(" ");
            let id = p.intern(name);
            if !p.is_atom_type[id as usize] {
                p.is_atom_type[id as usize] = true;
                p.atom_types.push(AtomTypeInfo { name: name.to_string(), mass, description });
            }
        }
        // list of hydrophilic types
        lines.next();
        // bonds
        for l in lines.by_ref() {
            if l.trim().is_empty() {
                break;
            }
            let Some(v) = nums(l, 5, 2) else { continue };
            let (a, b) = (p.intern(name_at(l, 0)), p.intern(name_at(l, 1)));
            p.bonds.entry((a.min(b), a.max(b))).or_insert(BondParam { k: v[0], r0: v[1] });
        }
        // angles
        for l in lines.by_ref() {
            if l.trim().is_empty() {
                break;
            }
            let Some(v) = nums(l, 8, 2) else { continue };
            let (a, b, c) = (p.intern(name_at(l, 0)), p.intern(name_at(l, 1)), p.intern(name_at(l, 2)));
            p.angles.entry((a.min(c), b, a.max(c))).or_insert(AngleParam { k: v[0], theta0: v[1] });
        }
        // torsions
        let mut last: Option<([u16; 4], bool)> = None;
        let mut skip = false;
        for l in lines.by_ref() {
            if l.trim().is_empty() {
                break;
            }
            let Some(v) = nums(l, 11, 4) else { continue };
            let key = torsion_key([0, 1, 2, 3].map(|k| p.intern(name_at(l, k))));
            let continued = last.is_some_and(|(k, more)| more && k == key);
            if !continued {
                // the first block of a torsion wins
                skip = p.torsions.contains_key(&key);
                if !skip {
                    p.torsions.insert(key, Vec::new());
                }
            }
            if !skip {
                let div = if v[0] > 0.0 { v[0] } else { 1.0 };
                let term = TorsionTerm { k: v[1] / div, phase: v[2], n: v[3].abs().round() as u32 };
                p.torsions.get_mut(&key).expect("torsion block").push(term);
            }
            last = Some((key, v[3] < 0.0));
        }
        // impropers
        for l in lines.by_ref() {
            if l.trim().is_empty() {
                break;
            }
            let Some(v) = nums(l, 11, 3) else { continue };
            let key = [0, 1, 2, 3].map(|k| p.intern(name_at(l, k)));
            p.impropers.push((key, TorsionTerm { k: v[0], phase: v[1], n: v[2].abs().round() as u32 }));
        }
        // van der Waals: the block after the "MOD4 RE" line
        for l in lines.by_ref() {
            if l.starts_with("MOD4") {
                break;
            }
        }
        for l in lines.by_ref() {
            if l.trim().is_empty() || l.starts_with("END") {
                break;
            }
            let mut it = l.split_whitespace();
            let Some(name) = it.next() else { continue };
            let (Some(r), Some(e)) = (it.next().and_then(|t| t.parse().ok()), it.next().and_then(|t| t.parse().ok())) else {
                continue;
            };
            let id = p.intern(name);
            p.vdw.entry(id).or_insert(VdwParam { r_star: r, eps: e });
        }
        p.angle_scale = p.fit_angle_scale();
        p
    }

    /// Median ratio of the tabulated angle force constants to eq. (5) of
    /// the paper (see [`estimate::angle_force_constant`]).
    fn fit_angle_scale(&self) -> f64 {
        let mut ratios: Vec<f64> = Vec::new();
        for (&(a, b, c), ap) in &self.angles {
            if ap.k <= 0.0 {
                continue;
            }
            let (Some(b1), Some(b2)) = (self.bond(a, b), self.bond(b, c)) else { continue };
            let el = |t: u16| type_element(&self.names[t as usize]);
            let (Some(za), Some(zb), Some(zc)) = (el(a), el(b), el(c)) else { continue };
            if let Some(k) = estimate::angle_force_constant(za, zb, zc, b1.r0, b2.r0, ap.theta0, 1.0) {
                ratios.push(ap.k / k);
            }
        }
        if ratios.is_empty() {
            return 1.0;
        }
        ratios.sort_by(|x, y| x.total_cmp(y));
        ratios[ratios.len() / 2]
    }

    /// Atom types of the file (name, mass, description), in file order.
    pub fn atom_types(&self) -> &[AtomTypeInfo] {
        &self.atom_types
    }
    /// Whether `name` is an atom type of this file.
    pub fn has_type(&self, name: &str) -> bool {
        self.ids.get(name).is_some_and(|&i| self.is_atom_type[i as usize])
    }
    pub(crate) fn id(&self, name: &str) -> Option<u16> {
        self.ids.get(name).copied()
    }
    pub(crate) fn name(&self, id: u16) -> &str {
        if id == WILD { "X" } else { &self.names[id as usize] }
    }
    pub(crate) fn bond(&self, a: u16, b: u16) -> Option<BondParam> {
        self.bonds.get(&(a.min(b), a.max(b))).copied()
    }
    pub(crate) fn angle(&self, a: u16, b: u16, c: u16) -> Option<AngleParam> {
        self.angles.get(&(a.min(c), b, a.max(c))).copied()
    }
    /// Terms of the torsion with exactly these types (`WILD` allowed).
    pub(crate) fn torsion(&self, key: [u16; 4]) -> Option<&[TorsionTerm]> {
        self.torsions.get(&torsion_key(key)).map(Vec::as_slice)
    }
    pub(crate) fn impropers(&self) -> &[([u16; 4], TorsionTerm)] {
        &self.impropers
    }
    pub(crate) fn vdw(&self, t: u16) -> Option<VdwParam> {
        self.vdw.get(&t).copied()
    }

    /// Bond parameters by type names.
    pub fn bond_param(&self, a: &str, b: &str) -> Option<BondParam> {
        self.bond(self.id(a)?, self.id(b)?)
    }
    /// Angle parameters by type names.
    pub fn angle_param(&self, a: &str, b: &str, c: &str) -> Option<AngleParam> {
        self.angle(self.id(a)?, self.id(b)?, self.id(c)?)
    }
    /// Torsion terms by type names (`"X"` = wildcard; exact entry only).
    pub fn torsion_param(&self, a: &str, b: &str, c: &str, d: &str) -> Option<&[TorsionTerm]> {
        let id = |n: &str| if n == "X" { Some(WILD) } else { self.id(n) };
        self.torsion([id(a)?, id(b)?, id(c)?, id(d)?])
    }
    /// Van der Waals parameters by type name.
    pub fn vdw_param(&self, t: &str) -> Option<VdwParam> {
        self.vdw(self.id(t)?)
    }
}
