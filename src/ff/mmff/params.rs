//! MMFF94 / MMFF94s parameter tables.
//!
//! Port of RDKit `Code/ForceField/MMFF/Params.h` and `Params.cpp`.
//! The tables themselves (`data/*.par`) were extracted verbatim from
//! `Params.cpp` with `tools/extract_mmff_params.py`; they are parsed with the
//! same rules RDKit uses and looked up with the same (binary-search based)
//! algorithms, so that parameter selection matches RDKit exactly.
//!
//! RDKit: Copyright (C) 2013-2024 Paolo Tosco and other RDKit contributors.
//! BSD 3-Clause license (see `LICENSE-RDKIT.txt`). The MMFF94 parameters are
//! Copyright (c) Merck and Co., Inc., 1994, 1995, 1996 (see the table headers).

use std::collections::HashMap;
use std::sync::OnceLock;

/// Atom type properties (MMFFPROP).
#[derive(Debug, Clone, Copy, Default)]
pub struct MmffProp {
    pub atno: u8,
    pub crd: u8,
    pub val: u8,
    pub pilp: u8,
    pub mltb: u8,
    pub arom: u8,
    pub linh: u8,
    pub sbmb: u8,
}

/// Partial bond charge increment and formal-charge adjustment factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct MmffPbci {
    pub pbci: f64,
    pub fcadj: f64,
}

/// Bond stretching parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MmffBond {
    pub kb: f64,
    pub r0: f64,
}

/// Herschbach–Laurie parameters for Badger's rule.
#[derive(Debug, Clone, Copy, Default)]
pub struct MmffHerschbachLaurie {
    pub a_ij: f64,
    pub d_ij: f64,
    pub dp_ij: f64,
}

/// Covalent radius and Pauling electronegativity.
#[derive(Debug, Clone, Copy, Default)]
pub struct MmffCovRadPauEle {
    pub r0: f64,
    pub chi: f64,
}

/// Angle bending parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MmffAngle {
    pub ka: f64,
    pub theta0: f64,
}

/// Stretch-bend parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MmffStbn {
    pub kba_ijk: f64,
    pub kba_kji: f64,
}

/// Out-of-plane bending parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MmffOop {
    pub koop: f64,
}

/// Torsion parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MmffTor {
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
}

/// Van der Waals parameters of one atom type.
#[derive(Debug, Clone, Copy, Default)]
pub struct MmffVdw {
    pub alpha_i: f64,
    pub n_i: f64,
    pub a_i: f64,
    pub g_i: f64,
    pub r_star: f64,
    pub da: u8,
}

/// Combined VdW parameters of an atom pair (RDKit `MMFFVdWRijstarEps`).
#[derive(Debug, Clone, Copy, Default)]
pub struct MmffVdwRijstarEps {
    pub r_ij_star_unscaled: f64,
    pub epsilon_unscaled: f64,
    pub r_ij_star: f64,
    pub epsilon: f64,
}

/// MMFF aromatic atom types (RDKit `defaultMMFFArom`).
pub const MMFF_AROMATIC_TYPES: [u8; 17] = [37, 38, 39, 44, 58, 59, 63, 64, 65, 66, 69, 76, 78, 79, 80, 81, 82];

// ---------------------------------------------------------------------------
// parsing helpers (RDKit: RDKit::getLine + boost::char_separator("\t"))
// ---------------------------------------------------------------------------

/// Data lines of a table: comment lines (`*`) skipped, tokens split on tabs
/// with empty tokens dropped (boost tokenizer default).
fn data_lines(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines()
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .filter(|l| !l.starts_with('*'))
        .map(|l| l.split('\t').filter(|t| !t.is_empty()).collect::<Vec<_>>())
}

fn pu(s: &str) -> u8 {
    s.trim().parse::<u32>().unwrap_or_else(|_| panic!("bad integer '{s}' in MMFF table")) as u8
}

fn pf(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or_else(|_| panic!("bad number '{s}' in MMFF table"))
}

/// `std::equal_range(keys[lo..hi], val)` (keys sorted within the range).
#[inline]
fn equal_range(keys: &[u8], lo: usize, hi: usize, val: u8) -> (usize, usize) {
    let s = &keys[lo..hi];
    let a = s.partition_point(|&k| k < val);
    let b = s.partition_point(|&k| k <= val);
    (lo + a, lo + b)
}

/// Nested equal_range over several key columns (in lookup order); returns
/// the index of the first matching row.
fn nested_lookup(cols: &[&[u8]], vals: &[u8]) -> Option<usize> {
    let (mut lo, mut hi) = (0, cols[0].len());
    for (c, &v) in cols.iter().zip(vals) {
        let (a, b) = equal_range(c, lo, hi, v);
        if a == b {
            return None;
        }
        lo = a;
        hi = b;
    }
    Some(lo)
}

// ---------------------------------------------------------------------------
// tables
// ---------------------------------------------------------------------------

/// Atom-type equivalence levels (MMFFDEF).
#[derive(Debug, Default)]
pub struct DefTable {
    params: Vec<[u8; 4]>,
}
impl DefTable {
    fn parse(text: &str) -> Self {
        let mut params = Vec::new();
        let mut old = 0u8;
        for t in data_lines(text) {
            let at = pu(t[1]);
            let lv = [pu(t[2]), pu(t[3]), pu(t[4]), pu(t[5])];
            if at != old {
                params.push(lv);
                old = at;
            }
        }
        DefTable { params }
    }
    /// Equivalence levels of `atom_type` (`None` for unknown types).
    pub fn get(&self, atom_type: u8) -> Option<&[u8; 4]> {
        if atom_type != 0 && (atom_type as usize) <= self.params.len() {
            Some(&self.params[atom_type as usize - 1])
        } else {
            None
        }
    }
    fn level(&self, atom_type: u8, l: usize) -> u8 {
        self.get(atom_type).map(|e| e[l]).unwrap_or(0)
    }
}

#[derive(Debug, Default)]
pub struct PropTable {
    types: Vec<u8>,
    params: Vec<MmffProp>,
}
impl PropTable {
    fn parse(text: &str) -> Self {
        let mut s = PropTable::default();
        for t in data_lines(text) {
            s.types.push(pu(t[0]));
            s.params.push(MmffProp {
                atno: pu(t[1]),
                crd: pu(t[2]),
                val: pu(t[3]),
                pilp: pu(t[4]),
                mltb: pu(t[5]),
                arom: pu(t[6]),
                linh: pu(t[7]),
                sbmb: pu(t[8]),
            });
        }
        s
    }
    pub fn get(&self, atom_type: u8) -> Option<&MmffProp> {
        nested_lookup(&[&self.types], &[atom_type]).map(|i| &self.params[i])
    }
}

#[derive(Debug, Default)]
pub struct PbciTable {
    params: Vec<MmffPbci>,
}
impl PbciTable {
    fn parse(text: &str) -> Self {
        let params = data_lines(text).map(|t| MmffPbci { pbci: pf(t[2]), fcadj: pf(t[3]) }).collect();
        PbciTable { params }
    }
    pub fn get(&self, atom_type: u8) -> Option<&MmffPbci> {
        if atom_type != 0 && (atom_type as usize) <= self.params.len() {
            Some(&self.params[atom_type as usize - 1])
        } else {
            None
        }
    }
}

#[derive(Debug, Default)]
pub struct ChgTable {
    bond_type: Vec<u8>,
    i: Vec<u8>,
    j: Vec<u8>,
    bci: Vec<f64>,
}
impl ChgTable {
    fn parse(text: &str) -> Self {
        let mut s = ChgTable::default();
        for t in data_lines(text) {
            s.bond_type.push(pu(t[0]));
            s.i.push(pu(t[1]));
            s.j.push(pu(t[2]));
            s.bci.push(pf(t[3]));
        }
        s
    }
    /// RDKit `getMMFFChgParams`: `(sign, bci)`.
    pub fn get(&self, bond_type: u8, i_type: u8, j_type: u8) -> (i32, Option<f64>) {
        let (mut ci, mut cj, mut sign) = (i_type, j_type, -1);
        if i_type > j_type {
            ci = j_type;
            cj = i_type;
            sign = 1;
        }
        let r = nested_lookup(&[&self.i, &self.j, &self.bond_type], &[ci, cj, bond_type]);
        (sign, r.map(|k| self.bci[k]))
    }
}

#[derive(Debug, Default)]
pub struct BondTable {
    bond_type: Vec<u8>,
    i: Vec<u8>,
    j: Vec<u8>,
    params: Vec<MmffBond>,
}
impl BondTable {
    fn parse(text: &str) -> Self {
        let mut s = BondTable::default();
        for t in data_lines(text) {
            s.bond_type.push(pu(t[0]));
            s.i.push(pu(t[1]));
            s.j.push(pu(t[2]));
            s.params.push(MmffBond { kb: pf(t[3]), r0: pf(t[4]) });
        }
        s
    }
    pub fn get(&self, bond_type: u8, a: u8, b: u8) -> Option<MmffBond> {
        let (ci, cj) = if a > b { (b, a) } else { (a, b) };
        nested_lookup(&[&self.i, &self.j, &self.bond_type], &[ci, cj, bond_type]).map(|k| self.params[k])
    }
}

#[derive(Debug, Default)]
pub struct BndkTable {
    i: Vec<u8>,
    j: Vec<u8>,
    params: Vec<MmffBond>,
}
impl BndkTable {
    fn parse(text: &str) -> Self {
        let mut s = BndkTable::default();
        for t in data_lines(text) {
            s.i.push(pu(t[0]));
            s.j.push(pu(t[1]));
            s.params.push(MmffBond { r0: pf(t[2]), kb: pf(t[3]) });
        }
        s
    }
    pub fn get(&self, a: u8, b: u8) -> Option<MmffBond> {
        let (ci, cj) = if a > b { (b, a) } else { (a, b) };
        nested_lookup(&[&self.i, &self.j], &[ci, cj]).map(|k| self.params[k])
    }
}

#[derive(Debug, Default)]
pub struct HerschbachLaurieTable {
    i: Vec<u8>,
    j: Vec<u8>,
    params: Vec<MmffHerschbachLaurie>,
}
impl HerschbachLaurieTable {
    fn parse(text: &str) -> Self {
        let mut s = HerschbachLaurieTable::default();
        for t in data_lines(text) {
            s.i.push(pu(t[0]));
            s.j.push(pu(t[1]));
            s.params.push(MmffHerschbachLaurie { a_ij: pf(t[2]), d_ij: pf(t[3]), dp_ij: pf(t[4]) });
        }
        s
    }
    pub fn get(&self, a: u8, b: u8) -> Option<MmffHerschbachLaurie> {
        let (ci, cj) = if a > b { (b, a) } else { (a, b) };
        nested_lookup(&[&self.i, &self.j], &[ci, cj]).map(|k| self.params[k])
    }
}

#[derive(Debug, Default)]
pub struct CovRadPauEleTable {
    anum: Vec<u8>,
    params: Vec<MmffCovRadPauEle>,
}
impl CovRadPauEleTable {
    fn parse(text: &str) -> Self {
        let mut s = CovRadPauEleTable::default();
        for t in data_lines(text) {
            s.anum.push(pu(t[0]));
            s.params.push(MmffCovRadPauEle { r0: pf(t[1]), chi: pf(t[2]) });
        }
        s
    }
    pub fn get(&self, anum: u8) -> Option<MmffCovRadPauEle> {
        nested_lookup(&[&self.anum], &[anum]).map(|k| self.params[k])
    }
}

#[derive(Debug, Default)]
pub struct AngleTable {
    angle_type: Vec<u8>,
    i: Vec<u8>,
    j: Vec<u8>,
    k: Vec<u8>,
    params: Vec<MmffAngle>,
}
impl AngleTable {
    fn parse(text: &str) -> Self {
        let mut s = AngleTable::default();
        for t in data_lines(text) {
            s.angle_type.push(pu(t[0]));
            s.i.push(pu(t[1]));
            s.j.push(pu(t[2]));
            s.k.push(pu(t[3]));
            s.params.push(MmffAngle { ka: pf(t[4]), theta0: pf(t[5]) });
        }
        s
    }
    /// RDKit `MMFFAngleCollection::operator()` (step-down through the
    /// equivalence levels of the outer atoms).
    pub fn get(&self, def: &DefTable, angle_type: u8, it: u8, jt: u8, kt: u8) -> Option<MmffAngle> {
        let (j_lo, j_hi) = equal_range(&self.j, 0, self.j.len(), jt);
        if j_lo == j_hi {
            return None;
        }
        for lvl in 0..4 {
            let mut ci = def.level(it, lvl);
            let mut ck = def.level(kt, lvl);
            if ci > ck {
                std::mem::swap(&mut ci, &mut ck);
            }
            let (a, b) = equal_range(&self.i, j_lo, j_hi, ci);
            if a == b {
                continue;
            }
            let (a, b) = equal_range(&self.k, a, b, ck);
            if a == b {
                continue;
            }
            let (a, b) = equal_range(&self.angle_type, a, b, angle_type);
            if a != b {
                return Some(self.params[a]);
            }
        }
        None
    }
}

#[derive(Debug, Default)]
pub struct StbnTable {
    stbn_type: Vec<u8>,
    i: Vec<u8>,
    j: Vec<u8>,
    k: Vec<u8>,
    params: Vec<MmffStbn>,
}
impl StbnTable {
    fn parse(text: &str) -> Self {
        let mut s = StbnTable::default();
        for t in data_lines(text) {
            s.stbn_type.push(pu(t[0]));
            s.i.push(pu(t[1]));
            s.j.push(pu(t[2]));
            s.k.push(pu(t[3]));
            s.params.push(MmffStbn { kba_ijk: pf(t[4]), kba_kji: pf(t[5]) });
        }
        s
    }
    /// RDKit `getMMFFStbnParams`: `(swap, params)`.
    pub fn get(&self, stbn_type: u8, bond_type1: u8, bond_type2: u8, it: u8, jt: u8, kt: u8) -> (bool, Option<MmffStbn>) {
        let (mut ci, mut ck, mut swap) = (it, kt, false);
        if it > kt {
            ci = kt;
            ck = it;
            swap = true;
        } else if it == kt {
            swap = bond_type1 < bond_type2;
        }
        let r = nested_lookup(&[&self.j, &self.i, &self.k, &self.stbn_type], &[jt, ci, ck, stbn_type]);
        (swap, r.map(|x| self.params[x]))
    }
}

#[derive(Debug, Default)]
pub struct DfsbTable {
    params: HashMap<(u8, u8, u8), MmffStbn>,
}
impl DfsbTable {
    fn parse(text: &str) -> Self {
        let mut params = HashMap::new();
        for t in data_lines(text) {
            params.insert((pu(t[0]), pu(t[1]), pu(t[2])), MmffStbn { kba_ijk: pf(t[3]), kba_kji: pf(t[4]) });
        }
        DfsbTable { params }
    }
    /// RDKit `getMMFFDfsbParams`: `(swap, params)`.
    pub fn get(&self, r1: u8, r2: u8, r3: u8) -> (bool, Option<MmffStbn>) {
        let (mut c1, mut c3, mut swap) = (r1, r3, false);
        if r1 > r3 {
            c1 = r3;
            c3 = r1;
            swap = true;
        }
        (swap, self.params.get(&(c1, r2, c3)).copied())
    }
}

#[derive(Debug, Default)]
pub struct OopTable {
    i: Vec<u8>,
    j: Vec<u8>,
    k: Vec<u8>,
    l: Vec<u8>,
    params: Vec<MmffOop>,
}
impl OopTable {
    fn parse(text: &str) -> Self {
        let mut s = OopTable::default();
        for t in data_lines(text) {
            s.i.push(pu(t[0]));
            s.j.push(pu(t[1]));
            s.k.push(pu(t[2]));
            s.l.push(pu(t[3]));
            s.params.push(MmffOop { koop: pf(t[4]) });
        }
        s
    }
    /// RDKit `MMFFOopCollection::operator()`.
    pub fn get(&self, def: &DefTable, it: u8, jt: u8, kt: u8, lt: u8) -> Option<MmffOop> {
        let (j_lo, j_hi) = equal_range(&self.j, 0, self.j.len(), jt);
        if j_lo == j_hi {
            return None;
        }
        for lvl in 0..4 {
            let mut c = [def.level(it, lvl), def.level(kt, lvl), def.level(lt, lvl)];
            c.sort_unstable();
            let (a, b) = equal_range(&self.i, j_lo, j_hi, c[0]);
            if a == b {
                continue;
            }
            let (a, b) = equal_range(&self.k, a, b, c[1]);
            if a == b {
                continue;
            }
            let (a, b) = equal_range(&self.l, a, b, c[2]);
            if a != b {
                return Some(self.params[a]);
            }
        }
        None
    }
}

#[derive(Debug, Default)]
pub struct TorTable {
    tor_type: Vec<u8>,
    i: Vec<u8>,
    j: Vec<u8>,
    k: Vec<u8>,
    l: Vec<u8>,
    params: Vec<MmffTor>,
}
impl TorTable {
    fn parse(text: &str) -> Self {
        let mut s = TorTable::default();
        for t in data_lines(text) {
            s.tor_type.push(pu(t[0]));
            s.i.push(pu(t[1]));
            s.j.push(pu(t[2]));
            s.k.push(pu(t[3]));
            s.l.push(pu(t[4]));
            s.params.push(MmffTor { v1: pf(t[5]), v2: pf(t[6]), v3: pf(t[7]) });
        }
        s
    }

    fn find(&self, tt: u8, ci: u8, cj: u8, ck: u8, cl: u8) -> Option<MmffTor> {
        nested_lookup(&[&self.j, &self.k, &self.i, &self.l, &self.tor_type], &[cj, ck, ci, cl, tt]).map(|x| self.params[x])
    }

    /// RDKit `getMMFFTorParams` (including its wildcard step-down and the
    /// fall-back to the secondary torsion type). Returns the torsion type
    /// that was used and the parameters.
    pub fn get(&self, def: &DefTable, tor_type: (u8, u8), it: u8, jt: u8, kt: u8, lt: u8) -> (u8, Option<MmffTor>) {
        let mut res: Option<MmffTor> = None;
        let mut iter: u32 = 0;
        let mut max_iter: u32 = 5;
        let mut can_tor_type = tor_type.0;
        while ((iter < max_iter) && (res.is_none() || max_iter == 4))
            || ((iter == 4) && (tor_type.0 == 5) && tor_type.1 != 0)
        {
            if max_iter == 5 && iter == 4 {
                max_iter = 4;
                iter = 0;
                can_tor_type = tor_type.1;
            }
            let (iw, lw) = match iter {
                1 => (1usize, 3usize),
                2 => (3, 1),
                x => (x as usize, x as usize),
            };
            let mut ci = def.level(it, iw);
            let mut cj = jt;
            let mut ck = kt;
            let mut cl = def.level(lt, lw);
            if cj > ck {
                std::mem::swap(&mut cj, &mut ck);
                std::mem::swap(&mut ci, &mut cl);
            } else if cj == ck && ci > cl {
                std::mem::swap(&mut ci, &mut cl);
            }
            if let Some(p) = self.find(can_tor_type, ci, cj, ck, cl) {
                res = Some(p);
                if max_iter == 4 {
                    break;
                }
            }
            iter += 1;
        }
        (can_tor_type, res)
    }
}

#[derive(Debug, Default)]
pub struct VdwTable {
    pub power: f64,
    pub b: f64,
    pub beta: f64,
    pub darad: f64,
    pub daeps: f64,
    types: Vec<u8>,
    params: Vec<MmffVdw>,
}
impl VdwTable {
    fn parse(text: &str) -> Self {
        let mut s = VdwTable::default();
        let mut first = true;
        for t in data_lines(text) {
            if first {
                first = false;
                s.power = pf(t[0]);
                s.b = pf(t[1]);
                s.beta = pf(t[2]);
                s.darad = pf(t[3]);
                s.daeps = pf(t[4]);
                continue;
            }
            s.types.push(pu(t[0]));
            let alpha_i = pf(t[1]);
            let a_i = pf(t[3]);
            s.params.push(MmffVdw {
                alpha_i,
                n_i: pf(t[2]),
                a_i,
                g_i: pf(t[4]),
                da: t[5].as_bytes()[0],
                r_star: a_i * alpha_i.powf(s.power),
            });
        }
        s
    }
    pub fn get(&self, atom_type: u8) -> Option<&MmffVdw> {
        nested_lookup(&[&self.types], &[atom_type]).map(|i| &self.params[i])
    }
}

/// All MMFF parameter tables of one variant.
#[derive(Debug)]
pub struct MmffParams {
    pub def: DefTable,
    pub prop: PropTable,
    pub pbci: PbciTable,
    pub chg: ChgTable,
    pub bond: BondTable,
    pub bndk: BndkTable,
    pub herschbach_laurie: HerschbachLaurieTable,
    pub cov_rad_pau_ele: CovRadPauEleTable,
    pub angle: AngleTable,
    pub stbn: StbnTable,
    pub dfsb: DfsbTable,
    pub oop: OopTable,
    pub tor: TorTable,
    pub vdw: VdwTable,
}

impl MmffParams {
    fn load(mmffs: bool) -> Self {
        MmffParams {
            def: DefTable::parse(include_str!("data/def.par")),
            prop: PropTable::parse(include_str!("data/prop.par")),
            pbci: PbciTable::parse(include_str!("data/pbci.par")),
            chg: ChgTable::parse(include_str!("data/chg.par")),
            bond: BondTable::parse(include_str!("data/bond.par")),
            bndk: BndkTable::parse(include_str!("data/bndk.par")),
            herschbach_laurie: HerschbachLaurieTable::parse(include_str!("data/herschbach_laurie.par")),
            cov_rad_pau_ele: CovRadPauEleTable::parse(include_str!("data/covrad_pauele.par")),
            angle: AngleTable::parse(include_str!("data/angle.par")),
            stbn: StbnTable::parse(include_str!("data/stbn.par")),
            dfsb: DfsbTable::parse(include_str!("data/dfsb.par")),
            oop: OopTable::parse(if mmffs { include_str!("data/oop_mmffs.par") } else { include_str!("data/oop.par") }),
            tor: TorTable::parse(if mmffs { include_str!("data/tor_mmffs.par") } else { include_str!("data/tor.par") }),
            vdw: VdwTable::parse(include_str!("data/vdw.par")),
        }
    }

    /// The (lazily parsed, shared) parameter set of a variant.
    pub fn get(mmffs: bool) -> &'static MmffParams {
        static MMFF94: OnceLock<MmffParams> = OnceLock::new();
        static MMFF94S: OnceLock<MmffParams> = OnceLock::new();
        if mmffs { MMFF94S.get_or_init(|| MmffParams::load(true)) } else { MMFF94.get_or_init(|| MmffParams::load(false)) }
    }
}

/// `true` if `atom_type` is an MMFF aromatic type.
pub fn is_mmff_aromatic(atom_type: u8) -> bool {
    MMFF_AROMATIC_TYPES.contains(&atom_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_by(cols: &[&[u8]]) -> bool {
        let n = cols[0].len();
        (1..n).all(|r| {
            let a: Vec<u8> = cols.iter().map(|c| c[r - 1]).collect();
            let b: Vec<u8> = cols.iter().map(|c| c[r]).collect();
            a <= b
        })
    }

    /// RDKit looks parameters up with nested binary searches, which is only
    /// equivalent to an exact-key lookup if the tables are sorted in the
    /// order of the search keys. Verify that for every table.
    #[test]
    fn tables_sorted_for_binary_search() {
        for mmffs in [false, true] {
            let p = MmffParams::get(mmffs);
            assert!(sorted_by(&[&p.prop.types]));
            assert!(sorted_by(&[&p.chg.i, &p.chg.j, &p.chg.bond_type]));
            assert!(sorted_by(&[&p.bond.i, &p.bond.j, &p.bond.bond_type]));
            assert!(sorted_by(&[&p.bndk.i, &p.bndk.j]));
            assert!(sorted_by(&[&p.herschbach_laurie.i, &p.herschbach_laurie.j]));
            assert!(sorted_by(&[&p.cov_rad_pau_ele.anum]));
            assert!(sorted_by(&[&p.angle.j, &p.angle.i, &p.angle.k, &p.angle.angle_type]));
            assert!(sorted_by(&[&p.stbn.j, &p.stbn.i, &p.stbn.k, &p.stbn.stbn_type]));
            assert!(sorted_by(&[&p.oop.j, &p.oop.i, &p.oop.k, &p.oop.l]));
            assert!(sorted_by(&[&p.tor.j, &p.tor.k, &p.tor.i, &p.tor.l, &p.tor.tor_type]));
            assert!(sorted_by(&[&p.vdw.types]));
        }
    }

    #[test]
    fn table_sizes_and_values() {
        let p = MmffParams::get(false);
        // types 83-86 do not exist; like RDKit, the MMFFDEF vector is
        // indexed by type-1 (only the ion types 87-99 are affected, and they
        // never take part in angle/torsion/oop lookups)
        assert_eq!(p.def.params.len(), 95);
        assert_eq!(p.pbci.params.len(), 99);
        let c = p.prop.get(1).unwrap();
        assert_eq!((c.atno, c.crd, c.val), (6, 4, 4));
        assert_eq!(p.bond.get(0, 1, 1), Some(MmffBond { kb: 4.258, r0: 1.508 }));
        assert_eq!(p.vdw.power, 0.25);
        assert_eq!(p.vdw.beta, 12.0);
        let v = p.vdw.get(5).unwrap();
        assert_eq!(v.da, b'-');
        // the MMFF94s tables differ from MMFF94 only in oop/torsions
        let s = MmffParams::get(true);
        assert_ne!(p.oop.params.len() + p.tor.params.len(), 0);
        assert_eq!(p.angle.params.len(), s.angle.params.len());
    }
}
