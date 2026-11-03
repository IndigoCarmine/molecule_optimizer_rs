//! MMFF94 atom typing, MMFF aromaticity, partial charges and parameter
//! assignment.
//!
//! Port of RDKit `Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp`
//! (`MMFFMolProperties`) and of `MolOps::setMMFFAromaticity` from
//! `Code/GraphMol/Aromaticity.cpp`.
//!
//! RDKit: Copyright (C) 2013-2021 Paolo Tosco, Copyright (C) 2004-2006
//! Rational Discovery LLC, and other RDKit contributors.
//! BSD 3-Clause license (see `LICENSE-RDKIT.txt`).

use super::MmffVariant;
use super::params::*;
use crate::core::{BondOrder, Molecule};
use crate::error::{Error, Result};
use crate::perception;
use crate::perception::rings::RingInfo;
use std::collections::{BTreeMap, BTreeSet};

/// RDKit `ForceFields::MMFF::isDoubleZero`.
#[inline]
pub(crate) fn is_double_zero(x: f64) -> bool {
    x < 1.0e-10 && x > -1.0e-10
}

/// Periodic table row starting from 0 for hydrogen (RDKit
/// `getPeriodicTableRow`).
pub(crate) fn periodic_table_row(anum: u8) -> u8 {
    match anum {
        3..=10 => 1,
        11..=18 => 2,
        19..=36 => 3,
        37..=54 => 4,
        _ => 0,
    }
}

/// Periodic table row for the Herschbach–Laurie table (RDKit
/// `getPeriodicTableRowHL`).
pub(crate) fn periodic_table_row_hl(anum: u8) -> u8 {
    let mut r = match anum {
        2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        _ => 0,
    };
    if (21..=30).contains(&anum) || (39..=48).contains(&anum) {
        r *= 10;
    }
    r
}

/// The molecule as seen by the MMFF typer: kekulized, with MMFF aromaticity
/// applied (aromatic ring bonds have order `Aromatic`), plus the explicit
/// valences cached before aromaticity perception (RDKit caches them during
/// sanitization).
#[derive(Debug, Clone)]
pub(crate) struct TypingMol {
    pub mol: Molecule,
    pub ri: RingInfo,
    /// explicit valence (RDKit `getValence(EXPLICIT)`)
    pub ev: Vec<i32>,
}

impl TypingMol {
    fn new(input: &Molecule) -> Result<TypingMol> {
        let mut mol = if input.atoms().iter().any(|a| a.is_aromatic) || input.bonds().iter().any(|b| b.order == BondOrder::Aromatic) {
            perception::kekulized_copy(input)?
        } else {
            input.clone()
        };
        let ri = match input.ring_info() {
            Some(r) => r.clone(),
            None => RingInfo::compute(&mol),
        };
        mol.ring_info = Some(ri.clone());
        let ev = (0..mol.num_atoms()).map(|i| mol.explicit_valence(i)).collect();
        let mut tm = TypingMol { mol, ri, ev };
        tm.set_mmff_aromaticity();
        Ok(tm)
    }

    #[inline]
    fn anum(&self, i: usize) -> u8 {
        self.mol.atom(i).atomic_num
    }
    #[inline]
    fn deg(&self, i: usize) -> usize {
        self.mol.degree(i)
    }
    #[inline]
    fn tdeg(&self, i: usize) -> usize {
        self.mol.total_degree(i)
    }
    #[inline]
    fn fc(&self, i: usize) -> i32 {
        self.mol.atom(i).formal_charge as i32
    }
    #[inline]
    fn arom(&self, i: usize) -> bool {
        self.mol.atom(i).is_aromatic
    }
    /// `getValence(EXPLICIT) + getNumImplicitHs()`
    #[inline]
    fn tval(&self, i: usize) -> i32 {
        self.ev[i] + self.mol.atom(i).implicit_hs as i32
    }
    #[inline]
    fn nbrs(&self, i: usize) -> Vec<usize> {
        self.mol.neighbors(i).collect()
    }
    /// Bond type between two bonded atoms.
    #[inline]
    fn bt(&self, a: usize, b: usize) -> BondOrder {
        self.mol.bond(self.mol.bond_between(a, b).expect("bonded atoms")).order
    }
    #[inline]
    fn in_ring_of_size(&self, a: usize, size: usize) -> bool {
        self.ri.is_atom_in_ring_of_size(a, size)
    }

    /// RDKit `isRingAromatic`: all bonds between consecutive ring atoms (the
    /// closing bond is not checked) are aromatic.
    fn is_ring_aromatic(&self, ring: &[usize]) -> bool {
        (0..ring.len().saturating_sub(1)).all(|i| self.bt(ring[i], ring[i + 1]) == BondOrder::Aromatic)
    }

    /// RDKit `MolOps::setMMFFAromaticity`.
    fn set_mmff_aromaticity(&mut self) {
        let rings = self.ri.atom_rings.clone();
        let n = self.mol.num_atoms();
        let mut arom_bit = vec![false; n];
        let mut arom_ring_bit = vec![false; rings.len()];
        let mut arom_rings_all_set = false;
        let mut n_arom_set: i64 = 0;
        let mut old_n_arom_set: i64 = -1;
        while !arom_rings_all_set && !rings.is_empty() && n_arom_set > old_n_arom_set {
            for (ri, ring) in rings.iter().enumerate() {
                let mut pi_e: u32 = 0;
                let mut move_to_next_ring = false;
                let mut is_nos_in_ring = false;
                let mut exo_double_bond = false;
                let mut j = 0;
                while !move_to_next_ring && j < ring.len() {
                    let a = ring[j];
                    let an = self.anum(a);
                    if an == 7 || an == 8 || (an == 16 && self.deg(a) == 2) {
                        is_nos_in_ring = true;
                    }
                    let next = if j == ring.len() - 1 { ring[0] } else { ring[j + 1] };
                    if self.bt(a, next) == BondOrder::Double {
                        pi_e += 2;
                    } else {
                        if an != 6 && !(an == 7 && self.tval(a) == 4) {
                            j += 1;
                            continue;
                        }
                        for nb in self.nbrs(a) {
                            if ring.contains(&nb) {
                                continue;
                            }
                            if self.bt(a, nb) == BondOrder::Single {
                                continue;
                            }
                            if self.ri.is_atom_in_ring(nb) && !arom_bit[nb] {
                                move_to_next_ring = true;
                                break;
                            }
                            if self.bt(a, nb) == BondOrder::Double {
                                if self.arom(nb) {
                                    pi_e += 1;
                                } else {
                                    exo_double_bond = true;
                                }
                            }
                        }
                    }
                    j += 1;
                }
                if move_to_next_ring {
                    continue;
                }
                let mut can_be_aromatic = true;
                for &a in ring {
                    arom_bit[a] = true;
                    let an = self.anum(a);
                    if (an == 6 || an == 7) && self.mol.atom(a).hybridization != crate::core::Hybridization::SP2 {
                        can_be_aromatic = false;
                    }
                }
                if !can_be_aromatic {
                    continue;
                }
                if is_nos_in_ring && !exo_double_bond && ring.len() % 2 == 1 {
                    pi_e += 2;
                }
                if pi_e > 2 && (pi_e - 2) % 4 == 0 {
                    arom_ring_bit[ri] = true;
                    for &a in ring {
                        self.mol.atom_mut(a).is_aromatic = true;
                    }
                }
            }
            old_n_arom_set = n_arom_set;
            n_arom_set = 0;
            arom_rings_all_set = true;
            for ring in &rings {
                for &a in ring {
                    if arom_bit[a] {
                        n_arom_set += 1;
                    } else {
                        arom_rings_all_set = false;
                    }
                }
            }
        }
        for (ri, ring) in rings.iter().enumerate() {
            if !arom_ring_bit[ri] {
                continue;
            }
            for j in 0..ring.len() {
                let next = if j == ring.len() - 1 { ring[0] } else { ring[j + 1] };
                let b = self.mol.bond_between(ring[j], next).expect("ring bond");
                let bond = self.mol.bond_mut(b);
                bond.order = BondOrder::Aromatic;
                bond.is_aromatic = true;
            }
        }
    }

    /// RDKit `isAtomNOxide`.
    fn is_n_oxide(&self, a: usize) -> bool {
        self.anum(a) == 7 && self.tdeg(a) >= 3 && self.nbrs(a).iter().any(|&n| self.anum(n) == 8 && self.tdeg(n) == 1)
    }
}

/// RDKit `isAngleInRingOfSize3or4`.
pub(crate) fn angle_in_ring_of_size_3_or_4(mol: &Molecule, i1: usize, i2: usize, i3: usize) -> u32 {
    if mol.bond_between(i1, i2).is_some() && mol.bond_between(i2, i3).is_some() {
        if mol.bond_between(i3, i1).is_some() {
            return 3;
        }
        let s1: BTreeSet<usize> = mol.neighbors(i1).filter(|&n| n != i2).collect();
        let s2: BTreeSet<usize> = mol.neighbors(i3).filter(|&n| n != i2).collect();
        if s1.intersection(&s2).next().is_some() {
            return 4;
        }
    }
    0
}

/// RDKit `isTorsionInRingOfSize4or5`.
pub(crate) fn torsion_in_ring_of_size_4_or_5(mol: &Molecule, i1: usize, i2: usize, i3: usize, i4: usize) -> u32 {
    if mol.bond_between(i1, i2).is_some() && mol.bond_between(i2, i3).is_some() && mol.bond_between(i3, i4).is_some() {
        if mol.bond_between(i4, i1).is_some() {
            return 4;
        }
        let s1: BTreeSet<usize> = mol.neighbors(i1).filter(|&n| n != i2).collect();
        let s2: BTreeSet<usize> = mol.neighbors(i4).filter(|&n| n != i3).collect();
        if s1.intersection(&s2).next().is_some() {
            return 5;
        }
    }
    0
}

/// RDKit `RingMembershipSize`.
struct RingMembershipSize {
    /// ring size -> atom -> (in aromatic ring of that size, ring ids)
    map: BTreeMap<usize, BTreeMap<usize, (bool, BTreeSet<u32>)>>,
}

const IS_AROMATIC_BIT: u32 = 1 << 31;

impl RingMembershipSize {
    fn new(tm: &TypingMol) -> Self {
        let mut map: BTreeMap<usize, BTreeMap<usize, (bool, BTreeSet<u32>)>> = BTreeMap::new();
        for (ri, ring) in tm.ri.atom_rings.iter().enumerate() {
            let arom = tm.is_ring_aromatic(ring);
            let id = ri as u32 | if arom { IS_AROMATIC_BIT } else { 0 };
            let m = map.entry(ring.len()).or_default();
            for &a in ring {
                let e = m.entry(a).or_default();
                e.1.insert(id);
                if arom {
                    e.0 = true;
                }
            }
        }
        RingMembershipSize { map }
    }
    fn is_atom_in_aromatic_ring_of_size(&self, a: usize, size: usize) -> bool {
        self.map.get(&size).and_then(|m| m.get(&a)).map(|e| e.0).unwrap_or(false)
    }
    fn are_atoms_in_same_ring_of_size(&self, size: usize, atoms: &[usize]) -> bool {
        let Some(m) = self.map.get(&size) else { return false };
        let Some(first) = m.get(&atoms[0]) else { return false };
        let mut common = first.1.clone();
        let mut res = false;
        for &a in &atoms[1..] {
            if common.is_empty() {
                break;
            }
            res = false;
            let Some(e) = m.get(&a) else { break };
            common = common.intersection(&e.1).copied().collect();
            res = !common.is_empty();
        }
        res
    }
}

/// MMFF atom types, formal and partial charges of a molecule (RDKit
/// `MMFFMolProperties`).
#[derive(Debug, Clone)]
pub struct MmffMolProperties {
    variant: MmffVariant,
    pub(crate) tm: TypingMol,
    atom_types: Vec<u8>,
    formal_charges: Vec<f64>,
    partial_charges: Vec<f64>,
    valid: bool,
    pub(crate) params: &'static MmffParams,
}

impl MmffMolProperties {
    /// Assigns MMFF atom types and charges. The molecule must have been
    /// sanitized (as done by the readers) and should have explicit
    /// hydrogens. If an atom cannot be typed, the result is not valid (see
    /// [`MmffMolProperties::is_valid`]) and charges are left at zero.
    pub fn new(mol: &Molecule, variant: MmffVariant) -> Result<Self> {
        if mol.num_atoms() == 0 {
            return Err(Error::ForceField("empty molecule".into()));
        }
        let tm = TypingMol::new(mol)?;
        let n = mol.num_atoms();
        let mut p = MmffMolProperties {
            variant,
            tm,
            atom_types: vec![0; n],
            formal_charges: vec![0.0; n],
            partial_charges: vec![0.0; n],
            valid: true,
            params: MmffParams::get(variant == MmffVariant::Mmff94s),
        };
        let rm = RingMembershipSize::new(&p.tm);
        for i in 0..n {
            if p.tm.anum(i) != 1 {
                p.set_heavy_atom_type(&rm, i);
            }
        }
        for i in 0..n {
            if p.tm.anum(i) == 1 {
                p.set_hydrogen_type(i);
            }
        }
        if p.valid {
            p.compute_charges();
        }
        Ok(p)
    }

    pub fn variant(&self) -> MmffVariant {
        self.variant
    }
    /// MMFF atom type of every atom (0 = untyped).
    pub fn atom_types(&self) -> &[u8] {
        &self.atom_types
    }
    /// MMFF partial charges.
    pub fn partial_charges(&self) -> &[f64] {
        &self.partial_charges
    }
    /// MMFF formal charges (fractional for delocalized groups).
    pub fn formal_charges(&self) -> &[f64] {
        &self.formal_charges
    }
    /// `true` if every atom received an MMFF atom type.
    pub fn is_valid(&self) -> bool {
        self.valid
    }
    /// The molecule as used for typing (kekulized, MMFF aromaticity).
    pub fn typing_molecule(&self) -> &Molecule {
        &self.tm.mol
    }

    #[inline]
    pub(crate) fn atom_type(&self, i: usize) -> u8 {
        self.atom_types[i]
    }

    fn prop(&self, t: u8) -> &MmffProp {
        self.params.prop.get(t).expect("MMFF properties for atom type")
    }

    // -----------------------------------------------------------------------
    // atom typing
    // -----------------------------------------------------------------------

    #[allow(clippy::cognitive_complexity)]
    fn set_heavy_atom_type(&mut self, rm: &RingMembershipSize, a: usize) {
        let tm = &self.tm;
        let mut atom_type: u8 = 0;
        let mut alpha_or_beta_in_same_ring = false;
        let mut is_nso2_or_nso3_or_ncn = false;
        let mut n_term_o_bonded_to_n = 0u32;
        let mut alpha_het: Vec<usize> = Vec::new();
        let mut beta_het: Vec<usize> = Vec::new();
        let anum = tm.anum(a);

        if tm.arom(a) {
            if rm.is_atom_in_aromatic_ring_of_size(a, 5) {
                if anum == 6 || anum == 7 {
                    for nb in tm.nbrs(a) {
                        if !rm.is_atom_in_aromatic_ring_of_size(nb, 5) {
                            continue;
                        }
                        let nba = tm.anum(nb);
                        if rm.are_atoms_in_same_ring_of_size(5, &[a, nb])
                            && (nba == 8 || nba == 16 || (nba == 7 && tm.tdeg(nb) == 3 && !tm.is_n_oxide(nb)))
                        {
                            alpha_het.push(nb);
                        }
                        for nb2 in tm.nbrs(nb) {
                            if nb2 == a {
                                continue;
                            }
                            if !rm.is_atom_in_aromatic_ring_of_size(nb2, 5) {
                                continue;
                            }
                            let nb2a = tm.anum(nb2);
                            if rm.are_atoms_in_same_ring_of_size(5, &[a, nb2])
                                && (nb2a == 8 || nb2a == 16 || (nb2a == 7 && tm.tdeg(nb2) == 3 && !tm.is_n_oxide(nb2)))
                            {
                                beta_het.push(nb2);
                            }
                        }
                    }
                    if !alpha_het.is_empty() && !beta_het.is_empty() {
                        'outer: for &x in &alpha_het {
                            for &y in &beta_het {
                                if rm.are_atoms_in_same_ring_of_size(5, &[x, y]) {
                                    alpha_or_beta_in_same_ring = true;
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
                let is_alpha_os = alpha_het.iter().any(|&x| tm.anum(x) == 8 || tm.anum(x) == 16);
                let is_beta_os = beta_het.iter().any(|&x| tm.anum(x) == 8 || tm.anum(x) == 16);

                match anum {
                    6 => 'c: {
                        if beta_het.is_empty() {
                            let (mut n_n, mut n_fc, mut n5, mut n6) = (0, 0, 0, 0);
                            for nb in tm.nbrs(a) {
                                if tm.anum(nb) == 7 && tm.tdeg(nb) == 3 {
                                    n_n += 1;
                                    if tm.fc(nb) > 0 && !tm.is_n_oxide(nb) {
                                        n_fc += 1;
                                    }
                                    if rm.is_atom_in_aromatic_ring_of_size(nb, 5) {
                                        n5 += 1;
                                    }
                                    if rm.is_atom_in_aromatic_ring_of_size(nb, 6) {
                                        n6 += 1;
                                    }
                                }
                            }
                            if (((n_n == 2) && n5 != 0) || ((n_n == 3) && (n5 == 2))) && n_fc != 0 && n6 == 0 {
                                // CIM+
                                atom_type = 80;
                                break 'c;
                            }
                        }
                        if alpha_het.len() == beta_het.len() {
                            let mut surrounded_by_benzene_c = true;
                            let mut surrounded_by_arom = true;
                            for nb in tm.nbrs(a) {
                                if tm.anum(nb) != 6 || !tm.in_ring_of_size(nb, 6) {
                                    surrounded_by_benzene_c = false;
                                }
                                if rm.are_atoms_in_same_ring_of_size(5, &[a, nb]) && !tm.arom(nb) {
                                    surrounded_by_arom = false;
                                }
                            }
                            if (alpha_het.is_empty() && beta_het.is_empty() && !surrounded_by_benzene_c && surrounded_by_arom)
                                || (!alpha_het.is_empty()
                                    && !beta_het.is_empty()
                                    && (!alpha_or_beta_in_same_ring || (!is_alpha_os && !is_beta_os)))
                            {
                                // C5
                                atom_type = 78;
                                break 'c;
                            }
                        }
                        if !alpha_het.is_empty() && (beta_het.is_empty() || is_alpha_os) {
                            // C5A
                            atom_type = 63;
                            break 'c;
                        }
                        if !beta_het.is_empty() && (alpha_het.is_empty() || is_beta_os) {
                            // C5B
                            atom_type = 64;
                            break 'c;
                        }
                    }
                    7 => 'n: {
                        if tm.is_n_oxide(a) {
                            // N5AX / N5BX / N5OX
                            atom_type = 82;
                            break 'n;
                        }
                        if alpha_het.is_empty() && beta_het.is_empty() {
                            if tm.tdeg(a) == 3 {
                                // NPYL
                                atom_type = 39;
                                break 'n;
                            }
                            // N5M
                            atom_type = 76;
                            break 'n;
                        }
                        if tm.tdeg(a) == 3 && alpha_het.len() != beta_het.len() {
                            // NIM+ / N5A+ / N5B+ / N5+
                            atom_type = 81;
                            break 'n;
                        }
                        if !alpha_het.is_empty() && (beta_het.is_empty() || is_alpha_os) {
                            // N5A
                            atom_type = 65;
                            break 'n;
                        }
                        if !beta_het.is_empty() && (alpha_het.is_empty() || is_beta_os) {
                            // N5B
                            atom_type = 66;
                            break 'n;
                        }
                        if !alpha_het.is_empty() && !beta_het.is_empty() {
                            // N5
                            atom_type = 79;
                            break 'n;
                        }
                    }
                    // OFUR
                    8 => atom_type = 59,
                    // STHI
                    16 => atom_type = 44,
                    _ => {}
                }
            }

            if atom_type == 0 && rm.is_atom_in_aromatic_ring_of_size(a, 6) {
                match anum {
                    // CB
                    6 => atom_type = 37,
                    7 => {
                        atom_type = if tm.is_n_oxide(a) {
                            // NPOX
                            69
                        } else if tm.tdeg(a) == 3 {
                            // NPD+
                            58
                        } else {
                            // NPYD
                            38
                        };
                    }
                    _ => {}
                }
            }
        }

        if atom_type == 0 {
            atom_type = match anum {
                3 => {
                    if tm.deg(a) == 0 { 92 } else { 0 }
                }
                6 => self.carbon_type(a),
                7 => {
                    // count terminal oxygens; detect sulfonamide/phosphonamide N
                    for nb in tm.nbrs(a) {
                        if tm.anum(nb) == 8 && tm.tdeg(nb) == 1 {
                            n_term_o_bonded_to_n += 1;
                        }
                        if tm.tval(a) >= 3 && (tm.anum(nb) == 15 || tm.anum(nb) == 16) {
                            let n_o = tm.nbrs(nb).iter().filter(|&&x| tm.anum(x) == 8 && tm.tdeg(x) == 1).count();
                            if !is_nso2_or_nso3_or_ncn {
                                is_nso2_or_nso3_or_ncn = n_o >= 2;
                            }
                        }
                    }
                    self.nitrogen_type(a, rm, n_term_o_bonded_to_n, is_nso2_or_nso3_or_ncn)
                }
                8 => self.oxygen_type(a),
                9 => match tm.deg(a) {
                    1 => 11,
                    0 => 89,
                    _ => 0,
                },
                11 => {
                    if tm.deg(a) == 0 { 93 } else { 0 }
                }
                12 => {
                    if tm.deg(a) == 0 { 99 } else { 0 }
                }
                14 => 19,
                15 => match tm.tdeg(a) {
                    4 => 25,
                    3 => 26,
                    2 => 75,
                    _ => 0,
                },
                16 => self.sulfur_type(a),
                17 => {
                    let mut t = 0;
                    if tm.tdeg(a) == 4 && tm.nbrs(a).iter().filter(|&&x| tm.anum(x) == 8).count() == 4 {
                        t = 77;
                    }
                    if t == 0 && tm.tdeg(a) == 1 {
                        t = 12;
                    }
                    if t == 0 && tm.deg(a) == 0 {
                        t = 90;
                    }
                    t
                }
                19 => {
                    if tm.deg(a) == 0 { 94 } else { 0 }
                }
                20 => {
                    if tm.deg(a) == 0 { 96 } else { 0 }
                }
                26 => {
                    if tm.deg(a) == 0 {
                        match tm.fc(a) {
                            2 => 87,
                            3 => 88,
                            _ => 0,
                        }
                    } else {
                        0
                    }
                }
                29 => {
                    if tm.deg(a) == 0 {
                        match tm.fc(a) {
                            1 => 97,
                            2 => 98,
                            _ => 0,
                        }
                    } else {
                        0
                    }
                }
                30 => {
                    if tm.deg(a) == 0 { 95 } else { 0 }
                }
                35 => match tm.deg(a) {
                    1 => 13,
                    0 => 91,
                    _ => 0,
                },
                53 => {
                    if tm.deg(a) == 1 { 14 } else { 0 }
                }
                _ => 0,
            };
        }
        self.atom_types[a] = atom_type;
        if atom_type == 0 {
            self.valid = false;
        }
    }

    fn carbon_type(&self, a: usize) -> u8 {
        let tm = &self.tm;
        match tm.tdeg(a) {
            4 => {
                if tm.in_ring_of_size(a, 3) {
                    // CR3R
                    22
                } else if tm.in_ring_of_size(a, 4) {
                    // CR4R
                    20
                } else {
                    // CR
                    1
                }
            }
            3 => {
                let (mut n_n2, mut n_n3, mut n_o, mut n_s) = (0, 0, 0, 0);
                let mut double_bonded_element = 0u8;
                for nb in tm.nbrs(a) {
                    if tm.bt(nb, a) == BondOrder::Double {
                        double_bonded_element = tm.anum(nb);
                    }
                    if tm.tdeg(nb) == 1 {
                        if tm.anum(nb) == 8 {
                            n_o += 1;
                        } else if tm.anum(nb) == 16 {
                            n_s += 1;
                        }
                    } else if tm.anum(nb) == 7 {
                        if tm.tdeg(nb) == 3 {
                            n_n3 += 1;
                        } else if tm.tdeg(nb) == 2 && tm.bt(nb, a) == BondOrder::Double {
                            n_n2 += 1;
                        }
                    }
                }
                if n_n3 >= 2 && n_n2 == 0 && double_bonded_element == 7 {
                    // CNN+ / CGD+
                    57
                } else if n_o == 2 || n_s == 2 {
                    // CO2M / CS2M
                    41
                } else if tm.in_ring_of_size(a, 4) && double_bonded_element == 6 {
                    // CR4E
                    30
                } else if matches!(double_bonded_element, 7 | 8 | 15 | 16) {
                    // C=N, C=O, ...
                    3
                } else {
                    // C=C / CSP2
                    2
                }
            }
            // CSP / =C=
            2 => 4,
            // C%-
            1 => 60,
            _ => 0,
        }
    }

    fn nitrogen_type(&self, a: usize, rm: &RingMembershipSize, n_term_o_bonded_to_n: u32, mut is_nso2_or_nso3_or_ncn: bool) -> u8 {
        let tm = &self.tm;
        if tm.tdeg(a) == 4 {
            // N3OX or NR+
            return if tm.is_n_oxide(a) { 68 } else { 34 };
        }
        if tm.tdeg(a) == 3 {
            if tm.tval(a) >= 4 {
                let mut double_bonded_cn = false;
                for nb in tm.nbrs(a) {
                    if tm.bt(nb, a) == BondOrder::Double {
                        double_bonded_cn = tm.anum(nb) == 7 || tm.anum(nb) == 6;
                        if tm.anum(nb) == 6 {
                            for nb2 in tm.nbrs(nb) {
                                if !double_bonded_cn {
                                    break;
                                }
                                if nb2 == a {
                                    continue;
                                }
                                double_bonded_cn = !(tm.anum(nb2) == 7 && tm.tdeg(nb2) == 3);
                            }
                        }
                    }
                }
                if n_term_o_bonded_to_n == 1 {
                    // N2OX
                    return 67;
                }
                if n_term_o_bonded_to_n >= 2 {
                    // NO2 / NO3
                    return 45;
                }
                if double_bonded_cn {
                    // N+=C / N+=N
                    return 54;
                }
            }
            if tm.tval(a) >= 3 {
                let mut is_nco_or_ncs = false;
                let mut is_ncn_plus = false;
                let mut is_ngd_plus = false;
                let mut is_nnn_or_nnc = false;
                let mut is_nbr_c = false;
                let mut is_nbr_benzene_c = false;
                let mut element_double_bonded_to_c = 0u8;
                let mut element_triple_bonded_to_c = 0u8;
                let mut n_o_bonded_to_c = 0;
                let mut n_s_bonded_to_c = 0;
                for nb in tm.nbrs(a) {
                    if tm.anum(nb) == 6 {
                        is_nbr_c = true;
                        if tm.arom(nb) && tm.in_ring_of_size(nb, 6) {
                            is_nbr_benzene_c = true;
                        }
                        let mut n_n2_bonded_to_c = 0;
                        let mut n_n3_bonded_to_c = 0;
                        n_o_bonded_to_c = 0;
                        n_s_bonded_to_c = 0;
                        let mut n_formal_charge = 0;
                        let mut n_in_aromatic_6_ring = 0;
                        for nb2 in tm.nbrs(nb) {
                            let bt = tm.bt(nb, nb2);
                            let nb2a = tm.anum(nb2);
                            if bt == BondOrder::Double && (nb2a == 8 || nb2a == 16) {
                                is_nco_or_ncs = true;
                            }
                            if bt == BondOrder::Double
                                || (bt == BondOrder::Aromatic && (nb2a == 6 || (nb2a == 7 && tm.ri.num_atom_rings(nb2) == 1)))
                            {
                                element_double_bonded_to_c = nb2a;
                            }
                            if bt == BondOrder::Triple {
                                element_triple_bonded_to_c = nb2a;
                            }
                            if nb2a == 7 && tm.tdeg(nb2) == 3 {
                                if tm.fc(nb2) == 1 {
                                    n_formal_charge += 1;
                                }
                                if rm.is_atom_in_aromatic_ring_of_size(nb, 6) {
                                    n_in_aromatic_6_ring += 1;
                                }
                                let n_o_bonded_to_n3 = tm.nbrs(nb2).iter().filter(|&&x| tm.anum(x) == 8).count();
                                if n_o_bonded_to_n3 < 2 {
                                    n_n3_bonded_to_c += 1;
                                }
                            }
                            if nb2a == 7 && tm.tdeg(nb2) == 2 && (bt == BondOrder::Double || bt == BondOrder::Aromatic) {
                                n_n2_bonded_to_c += 1;
                            }
                            if tm.arom(nb2) {
                                if nb2a == 8 {
                                    n_o_bonded_to_c += 1;
                                }
                                if nb2a == 16 {
                                    n_s_bonded_to_c += 1;
                                }
                            }
                        }
                        if element_double_bonded_to_c == 7 {
                            if n_n3_bonded_to_c == 2
                                && n_n2_bonded_to_c == 0
                                && n_formal_charge != 0
                                && n_in_aromatic_6_ring == 0
                                && tm.tdeg(nb) < 4
                            {
                                is_ncn_plus = true;
                            }
                            if n_n3_bonded_to_c == 3 {
                                is_ngd_plus = true;
                            }
                        }
                    }
                    if tm.anum(nb) == 7 {
                        let (mut n_n, mut n_o, mut n_s) = (0, 0, 0);
                        for nb2 in tm.nbrs(nb) {
                            if tm.bt(nb, nb2) == BondOrder::Double {
                                if tm.anum(nb2) == 6 {
                                    for nb3 in tm.nbrs(nb2) {
                                        if nb3 == nb {
                                            continue;
                                        }
                                        match tm.anum(nb3) {
                                            7 => n_n += 1,
                                            8 => n_o += 1,
                                            16 => n_s += 1,
                                            _ => {}
                                        }
                                    }
                                    if n_o == 0 && n_s == 0 && n_n == 0 && !is_nbr_benzene_c {
                                        is_nnn_or_nnc = true;
                                    }
                                }
                                if tm.anum(nb2) == 7 && !is_nbr_benzene_c {
                                    is_nnn_or_nnc = true;
                                }
                            }
                        }
                    }
                }
                if is_nbr_c {
                    if element_triple_bonded_to_c == 7 {
                        is_nso2_or_nso3_or_ncn = true;
                    }
                    if is_ncn_plus {
                        // NCN+
                        return 55;
                    }
                    if is_ngd_plus {
                        // NGD+
                        return 56;
                    }
                    if (!is_nco_or_ncs && !is_nso2_or_nso3_or_ncn)
                        && ((n_o_bonded_to_c == 0 && n_s_bonded_to_c == 0 && is_nbr_benzene_c)
                            || (element_double_bonded_to_c == 6
                                || element_double_bonded_to_c == 7
                                || element_double_bonded_to_c == 15
                                || element_triple_bonded_to_c == 6))
                    {
                        // NC=C / NC=N / NC=P / NC%C
                        return 40;
                    }
                }
                if !is_nso2_or_nso3_or_ncn && (is_nco_or_ncs || is_nnn_or_nnc) {
                    // NC=O / NC=S / NN=C / NN=N
                    return 10;
                }
            }
        }
        if tm.tdeg(a) == 2 {
            if tm.tval(a) == 4 {
                let is_isonitrile = tm.nbrs(a).iter().any(|&nb| tm.bt(a, nb) == BondOrder::Triple);
                // NR% or =N=
                return if is_isonitrile { 61 } else { 53 };
            }
            if tm.tval(a) == 3 {
                let mut is_nitroso = false;
                let mut is_imine_or_azo = false;
                for nb in tm.nbrs(a) {
                    if tm.bt(a, nb) == BondOrder::Double {
                        is_nitroso = tm.anum(nb) == 8 && n_term_o_bonded_to_n == 1;
                        is_imine_or_azo = tm.anum(nb) == 6 || tm.anum(nb) == 7;
                    }
                }
                if is_nitroso && !is_imine_or_azo {
                    // N=O
                    return 46;
                }
                if is_imine_or_azo {
                    // N=C / N=N
                    return 9;
                }
            }
            if tm.tval(a) >= 2 {
                let mut is_nso = false;
                for nb in tm.nbrs(a) {
                    if is_nso {
                        break;
                    }
                    if tm.anum(nb) == 16 {
                        let n_term_o = tm.nbrs(nb).iter().filter(|&&x| tm.anum(x) == 8 && tm.tdeg(x) == 1).count();
                        is_nso = n_term_o == 1;
                    }
                }
                if is_nso {
                    // NSO
                    return 48;
                }
                if !is_nso2_or_nso3_or_ncn {
                    // NM
                    return 62;
                }
            }
        }
        if is_nso2_or_nso3_or_ncn {
            // NSO2 / NSO3 / NC%N
            return 43;
        }
        if tm.tdeg(a) == 1 {
            let mut is_nsp = false;
            let mut is_nazt = false;
            for nb in tm.nbrs(a) {
                if is_nsp || is_nazt {
                    break;
                }
                is_nsp = tm.bt(a, nb) == BondOrder::Triple;
                if tm.anum(nb) == 7 && tm.tdeg(nb) == 2 {
                    for nb2 in tm.nbrs(nb) {
                        if is_nazt {
                            break;
                        }
                        is_nazt = (tm.anum(nb2) == 7 && tm.tdeg(nb2) == 2) || (tm.anum(nb2) == 6 && tm.tdeg(nb2) == 3);
                    }
                }
            }
            if is_nsp {
                // NSP
                return 42;
            }
            if is_nazt {
                // NAZT
                return 47;
            }
        }
        // NR
        8
    }

    fn oxygen_type(&self, a: usize) -> u8 {
        let tm = &self.tm;
        if tm.tdeg(a) == 3 {
            // O+
            return 49;
        }
        if tm.tdeg(a) == 2 {
            if tm.tval(a) == 3 {
                // O=+
                return 51;
            }
            let n_h = tm.nbrs(a).iter().filter(|&&x| tm.anum(x) == 1).count();
            if n_h + tm.mol.atom(a).implicit_hs as usize == 2 {
                // OH2
                return 70;
            }
            // OC=O, OC=C, OC=N, OC=S, ONO2, ON=O, OSO3, OSO2, OSO, OS=O, -OS,
            // OPO3, OPO2, OPO, -OP
            return 6;
        }
        if tm.deg(a) <= 1 {
            let mut n_n = 0;
            let mut n_o = 0;
            let mut n_s = 0;
            let at = tm.mol.atom(a);
            let mut is_oxide_o_bonded_to_h = at.num_explicit_hs as u32 + at.implicit_hs as u32 > 0;
            let mut is_carboxylate_o = false;
            let mut is_carbonyl_o = false;
            let mut is_oxide_o_bonded_to_c = false;
            let mut is_nitroso_o = false;
            let mut is_oxide_o_bonded_to_n = false;
            let mut is_n_oxide_o = false;
            let mut is_nitro_o = false;
            let mut is_thio_sulfinate_o = false;
            let mut is_sulfate_o = false;
            let mut is_sulfoxide_o = false;
            let mut is_phosphate_or_perchlorate_o = false;
            for nb in tm.nbrs(a) {
                if is_oxide_o_bonded_to_c
                    || is_oxide_o_bonded_to_n
                    || is_oxide_o_bonded_to_h
                    || is_carboxylate_o
                    || is_nitro_o
                    || is_n_oxide_o
                    || is_thio_sulfinate_o
                    || is_sulfate_o
                    || is_phosphate_or_perchlorate_o
                    || is_carbonyl_o
                    || is_nitroso_o
                    || is_sulfoxide_o
                {
                    break;
                }
                let bt = tm.bt(a, nb);
                let nba = tm.anum(nb);
                if nba == 6 || nba == 7 || nba == 16 {
                    for nb2 in tm.nbrs(nb) {
                        if tm.anum(nb2) == 7 && tm.tdeg(nb2) == 2 {
                            n_n += 1;
                        }
                        if tm.anum(nb2) == 8 && tm.tdeg(nb2) == 1 {
                            n_o += 1;
                        }
                        if tm.anum(nb2) == 16 && tm.tdeg(nb2) == 1 {
                            n_s += 1;
                        }
                    }
                }
                is_oxide_o_bonded_to_h = nba == 1;
                if nba == 6 {
                    is_carboxylate_o = n_o == 2;
                    is_carbonyl_o = bt == BondOrder::Double;
                    is_oxide_o_bonded_to_c = bt == BondOrder::Single && n_o == 1;
                }
                if nba == 7 {
                    is_nitroso_o = bt == BondOrder::Double;
                    if bt == BondOrder::Single && n_o == 1 {
                        is_oxide_o_bonded_to_n = tm.tdeg(nb) == 2 || tm.tval(nb) == 3;
                        is_n_oxide_o = tm.tval(nb) == 4;
                    }
                    is_nitro_o = n_o >= 2;
                }
                if nba == 16 {
                    is_thio_sulfinate_o = n_s == 1;
                    is_sulfate_o = bt == BondOrder::Single || (bt == BondOrder::Double && (n_o + n_n) > 1);
                    is_sulfoxide_o = bt == BondOrder::Double && (n_o + n_n) == 1;
                }
                is_phosphate_or_perchlorate_o = nba == 15 || nba == 17;
            }
            if is_oxide_o_bonded_to_c || is_oxide_o_bonded_to_n || is_oxide_o_bonded_to_h {
                // OM / OM2
                return 35;
            }
            if is_carboxylate_o || is_nitro_o || is_n_oxide_o || is_thio_sulfinate_o || is_sulfate_o || is_phosphate_or_perchlorate_o {
                // O2CM, ONX, O2N, O2NO, O3N, OSMS, O-S, O2S, O3S, O4S, OP, O2P,
                // O3P, O4P, O4Cl
                return 32;
            }
            if is_carbonyl_o || is_nitroso_o || is_sulfoxide_o {
                // O=C, O=N, O=S
                return 7;
            }
        }
        0
    }

    fn sulfur_type(&self, a: usize) -> u8 {
        let tm = &self.tm;
        let td = tm.tdeg(a);
        if td == 3 || td == 4 {
            let mut n_o_or_n = 0;
            let mut n_s = 0;
            let mut is_c_double_bonded_to_s = false;
            for nb in tm.nbrs(a) {
                if tm.anum(nb) == 6 && tm.bt(a, nb) == BondOrder::Double {
                    is_c_double_bonded_to_s = true;
                }
                if (tm.deg(nb) == 1 && tm.anum(nb) == 8) || (tm.tdeg(nb) == 2 && tm.anum(nb) == 7) {
                    n_o_or_n += 1;
                }
                if tm.deg(nb) == 1 && tm.anum(nb) == 16 {
                    n_s += 1;
                }
            }
            if (td == 3 && n_o_or_n == 2 && is_c_double_bonded_to_s) || td == 4 {
                // =SO2
                return 18;
            }
            if (n_o_or_n != 0 && n_s != 0) || (n_o_or_n == 2 && !is_c_double_bonded_to_s) {
                // SSOM
                return 73;
            }
            // S=O / >S=N
            return 17;
        }
        if td == 2 {
            let is_o_double = tm.nbrs(a).iter().any(|&nb| tm.anum(nb) == 8 && tm.bt(a, nb) == BondOrder::Double);
            // =S=O or S
            return if is_o_double { 74 } else { 15 };
        }
        if tm.deg(a) == 1 {
            let mut n_term_s = 0;
            let mut is_c_double_bonded_to_s = false;
            for nb in tm.nbrs(a) {
                for nb2 in tm.nbrs(nb) {
                    if tm.anum(nb2) == 16 && tm.tdeg(nb2) == 1 {
                        n_term_s += 1;
                    }
                }
                if tm.anum(nb) == 6 && tm.bt(a, nb) == BondOrder::Double {
                    is_c_double_bonded_to_s = true;
                }
            }
            if is_c_double_bonded_to_s && n_term_s != 2 {
                // S=C
                return 16;
            }
            // S-P / SM / SSMO
            return 72;
        }
        0
    }

    fn set_hydrogen_type(&mut self, a: usize) {
        let tm = &self.tm;
        let mut atom_type = 0u8;
        for nb in tm.nbrs(a) {
            match tm.anum(nb) {
                // HC / HSI
                6 | 14 => atom_type = 5,
                7 => {
                    atom_type = match self.atom_types[nb] {
                        // HNR, H3N, HPYL, HNOX
                        8 | 39 | 62 | 67 | 68 => 23,
                        // HNR+ ... HIM+
                        34 | 54 | 55 | 56 | 58 | 81 => 36,
                        // HN=N / HN=C
                        9 => 27,
                        // HNCC, HNCN, HNCO, ...
                        _ => 28,
                    }
                }
                8 => {
                    atom_type = match self.atom_types[nb] {
                        // HO+
                        49 => 50,
                        // HO=+
                        51 => 52,
                        // HOH
                        70 => 31,
                        6 => {
                            let (mut hocc, mut hoco, mut hop, mut hos) = (false, false, false, false);
                            for nb2 in tm.nbrs(nb) {
                                if tm.anum(nb2) == 6 {
                                    for nb3 in tm.nbrs(nb2) {
                                        if nb3 == nb {
                                            continue;
                                        }
                                        let bt = tm.bt(nb2, nb3);
                                        let a3 = tm.anum(nb3);
                                        if (a3 == 6 || a3 == 7) && (bt == BondOrder::Double || bt == BondOrder::Aromatic) {
                                            hocc = true;
                                        }
                                        if a3 == 8 && bt == BondOrder::Double {
                                            hoco = true;
                                        }
                                    }
                                }
                                if tm.anum(nb2) == 15 {
                                    hop = true;
                                }
                                if tm.anum(nb2) == 16 {
                                    hos = true;
                                }
                            }
                            if hoco || hop {
                                // HOCO
                                24
                            } else if hocc {
                                // HOCC / HOCN
                                29
                            } else if hos {
                                // HOS
                                33
                            } else {
                                // HO / HOR
                                21
                            }
                        }
                        _ => 21,
                    }
                }
                // HP / HS / HS=N
                15 | 16 => atom_type = 71,
                _ => {}
            }
        }
        self.atom_types[a] = atom_type;
        if atom_type == 0 {
            self.valid = false;
        }
    }

    // -----------------------------------------------------------------------
    // charges
    // -----------------------------------------------------------------------

    fn compute_charges(&mut self) {
        let tm = &self.tm;
        let n = tm.mol.num_atoms();
        let p = self.params;
        let mut fchg = vec![0.0f64; n];
        for idx in 0..n {
            let atom_type = self.atom_types[idx];
            let mut f = 0.0f64;
            match atom_type {
                32 | 72 => {
                    for nb in tm.nbrs(idx) {
                        let nbr_type = self.atom_types[nb];
                        let mut n_sec_n = 0i32;
                        let mut n_term_os = 0i32;
                        for nb2 in tm.nbrs(nb) {
                            if tm.anum(nb2) == 7 && tm.deg(nb2) == 2 && !tm.arom(nb2) {
                                n_sec_n += 1;
                            }
                            if (tm.anum(nb2) == 8 || tm.anum(nb2) == 16) && tm.deg(nb2) == 1 {
                                n_term_os += 1;
                            }
                        }
                        if tm.anum(nb) == 16 && n_term_os == 2 && n_sec_n == 1 {
                            n_sec_n = 0;
                        }
                        if tm.anum(nb) == 6 && n_term_os != 0 {
                            f = if n_term_os == 1 { -1.0 } else { -((n_term_os - 1) as f64 / n_term_os as f64) };
                            break;
                        }
                        if nbr_type == 45 && n_term_os == 3 {
                            f = -1.0 / 3.0;
                            break;
                        }
                        if nbr_type == 25 && n_term_os != 0 {
                            f = if n_term_os == 1 { 0.0 } else { -((n_term_os - 1) as f64 / n_term_os as f64) };
                            break;
                        }
                        if nbr_type == 18 && n_term_os != 0 {
                            f = if n_sec_n + n_term_os == 2 {
                                0.0
                            } else {
                                -(((n_sec_n + n_term_os) - 2) as f64 / n_term_os as f64)
                            };
                            break;
                        }
                        if nbr_type == 73 && n_term_os != 0 {
                            f = if n_term_os == 1 { 0.0 } else { -((n_term_os - 1) as f64 / n_term_os as f64) };
                            break;
                        }
                        if nbr_type == 77 && n_term_os != 0 {
                            f = -(1.0 / n_term_os as f64);
                            break;
                        }
                    }
                }
                76 => {
                    if let Some(ring) = tm.ri.atom_rings.iter().find(|r| r.contains(&idx)) {
                        let n76 = ring.iter().filter(|&&x| self.atom_types[x] == 76).count();
                        if n76 != 0 {
                            f = -(1.0 / n76 as f64);
                        }
                    }
                }
                55 | 56 | 81 => {
                    f = tm.fc(idx) as f64;
                    let mut n_conj = 1u32;
                    let mut old_n_conj = 0u32;
                    let mut conj = vec![false; n];
                    conj[idx] = true;
                    while n_conj > old_n_conj {
                        old_n_conj = n_conj;
                        for i in 0..n {
                            if !conj[i] {
                                continue;
                            }
                            for nb in tm.nbrs(i) {
                                let t = self.atom_types[nb];
                                if t != 57 && t != 80 {
                                    continue;
                                }
                                for nb2 in tm.nbrs(nb) {
                                    let t2 = self.atom_types[nb2];
                                    if t2 != 55 && t2 != 56 && t2 != 81 {
                                        continue;
                                    }
                                    if !conj[nb2] {
                                        conj[nb2] = true;
                                        f += tm.fc(nb2) as f64;
                                        n_conj += 1;
                                    }
                                }
                            }
                        }
                    }
                    if n_conj != 0 {
                        f /= n_conj as f64;
                    }
                }
                61 => {
                    for nb in tm.nbrs(idx) {
                        if self.atom_types[nb] == 42 {
                            f = 1.0;
                        }
                    }
                }
                34 | 49 | 51 | 54 | 58 | 92 | 93 | 94 | 97 => f = 1.0,
                87 | 95 | 96 | 98 | 99 => f = 2.0,
                88 => f = 3.0,
                35 | 62 | 89 | 90 | 91 => f = -1.0,
                _ => {}
            }
            fchg[idx] = f;
        }
        let mut pchg = vec![0.0f64; n];
        for idx in 0..n {
            let atom_type = self.atom_types[idx];
            let mut q0 = fchg[idx];
            let m = p.prop.get(atom_type).expect("prop").crd as f64;
            let v = p.pbci.get(atom_type).expect("pbci").fcadj;
            let mut sum_formal = 0.0;
            let mut sum_partial = 0.0;
            if is_double_zero(v) {
                for nb in tm.nbrs(idx) {
                    let nf = fchg[nb];
                    if nf < 0.0 {
                        q0 += nf / (2.0 * tm.deg(nb) as f64);
                    }
                }
            }
            if atom_type == 62 {
                for nb in tm.nbrs(idx) {
                    let nf = fchg[nb];
                    if nf > 0.0 {
                        q0 -= nf / 2.0;
                    }
                }
            }
            for nb in tm.nbrs(idx) {
                let nbr_type = self.atom_types[nb];
                let bond_type = self.bond_type(idx, nb);
                let (sign, bci) = p.chg.get(bond_type, atom_type, nbr_type);
                sum_partial += match bci {
                    Some(b) => sign as f64 * b,
                    None => p.pbci.get(atom_type).expect("pbci").pbci - p.pbci.get(nbr_type).expect("pbci").pbci,
                };
                sum_formal += fchg[nb];
            }
            pchg[idx] = (1.0 - m * v) * q0 + v * sum_formal + sum_partial;
        }
        self.formal_charges = fchg;
        self.partial_charges = pchg;
    }

    // -----------------------------------------------------------------------
    // parameters
    // -----------------------------------------------------------------------

    /// RDKit `getMMFFBondType`.
    pub(crate) fn bond_type(&self, a: usize, b: usize) -> u8 {
        let bi = self.tm.mol.bond_between(a, b).expect("bond");
        let bond = self.tm.mol.bond(bi);
        let p1 = self.prop(self.atom_types[bond.begin]);
        let p2 = self.prop(self.atom_types[bond.end]);
        u8::from(bond.order == BondOrder::Single && ((p1.sbmb != 0 && p2.sbmb != 0) || (p1.arom != 0 && p2.arom != 0)))
    }

    /// RDKit `getMMFFAngleType`.
    pub(crate) fn angle_type(&self, i1: usize, i2: usize, i3: usize) -> u8 {
        let sum = self.bond_type(i1, i2) as u32 + self.bond_type(i2, i3) as u32;
        let mut angle_type = sum;
        let size = angle_in_ring_of_size_3_or_4(&self.tm.mol, i1, i2, i3);
        if size != 0 {
            angle_type = size;
            if sum != 0 {
                angle_type += sum + size - 2;
            }
        }
        angle_type as u8
    }

    /// RDKit `getMMFFTorsionType`.
    pub(crate) fn torsion_type(&self, i1: usize, i2: usize, i3: usize, i4: usize) -> (u8, u8) {
        let mol = &self.tm.mol;
        let bjk = mol.bond(mol.bond_between(i2, i3).expect("bond"));
        let bt_ij = self.bond_type(i1, i2);
        let bt_jk = self.bond_type(i2, i3);
        let bt_kl = self.bond_type(i3, i4);
        let mut tor_type = bt_jk;
        let mut second = 0u8;
        if bt_jk == 0 && bjk.order == BondOrder::Single && (bt_ij == 1 || bt_kl == 1) {
            tor_type = 2;
        }
        let size = torsion_in_ring_of_size_4_or_5(mol, i1, i2, i3, i4);
        if size == 4 && !(mol.bond_between(i1, i3).is_some() || mol.bond_between(i2, i4).is_some()) {
            second = tor_type;
            tor_type = 4;
        } else if size == 5 && [i1, i2, i3, i4].iter().any(|&x| self.atom_types[x] == 1) {
            second = tor_type;
            tor_type = 5;
        }
        (tor_type, second)
    }

    /// RDKit `getMMFFBondStretchEmpiricalRuleParams`.
    fn bond_stretch_empirical(&self, a: usize, b: usize) -> MmffBond {
        let p = self.params;
        let bond = self.tm.mol.bond(self.tm.mol.bond_between(a, b).expect("bond"));
        let an1 = self.tm.anum(bond.begin);
        let an2 = self.tm.anum(bond.end);
        let bndk = p.bndk.get(an1, an2);
        let c1 = p.cov_rad_pau_ele.get(an1).expect("covalent radius/Pauling electronegativity parameters for atom 1 not found");
        let c2 = p.cov_rad_pau_ele.get(an2).expect("covalent radius/Pauling electronegativity parameters for atom 2 not found");
        let c = if an1 == 1 || an2 == 1 { 0.050 } else { 0.085 };
        let n = 1.4;
        let delta = 0.0;
        let r0 = c1.r0 + c2.r0 - c * (c1.chi - c2.chi).abs().powf(n) - delta;
        let kb = if let Some(bk) = bndk {
            let coeff = bk.r0 / r0;
            let coeff2 = coeff * coeff;
            let coeff6 = coeff2 * coeff2 * coeff2;
            bk.kb * coeff6
        } else {
            let hl = p
                .herschbach_laurie
                .get(periodic_table_row_hl(an1), periodic_table_row_hl(an2))
                .expect("Herschbach-Laurie parameters");
            10.0f64.powf(-(r0 - hl.a_ij) / hl.d_ij)
        };
        MmffBond { kb, r0 }
    }

    /// RDKit `getMMFFBondStretchParams`: `(bondType, params)`.
    pub fn bond_stretch_params(&self, a: usize, b: usize) -> Option<(u8, MmffBond)> {
        if !self.valid || self.tm.mol.bond_between(a, b).is_none() {
            return None;
        }
        let bt = self.bond_type(a, b);
        let p = match self.params.bond.get(bt, self.atom_types[a], self.atom_types[b]) {
            Some(p) => p,
            None => self.bond_stretch_empirical(a, b),
        };
        Some((bt, p))
    }

    /// RDKit `getMMFFAngleBendEmpiricalRuleParams`.
    fn angle_bend_empirical(
        &self,
        old: Option<MmffAngle>,
        central: &MmffProp,
        b1: &MmffBond,
        b2: &MmffBond,
        idx: [usize; 3],
    ) -> MmffAngle {
        let an = [self.tm.anum(idx[0]), self.tm.anum(idx[1]), self.tm.anum(idx[2])];
        let ring_size = angle_in_ring_of_size_3_or_4(&self.tm.mol, idx[0], idx[1], idx[2]);
        let theta0 = match old {
            Some(o) => o.theta0,
            None => {
                let mut t = 120.0;
                match central.crd {
                    4 => t = 109.45,
                    2 => {
                        if an[1] == 8 {
                            t = 105.0;
                        } else if central.linh == 1 {
                            t = 180.0;
                        }
                    }
                    3 => {
                        if central.val == 3 && central.mltb == 0 {
                            t = if an[1] == 7 { 107.0 } else { 92.0 };
                        }
                    }
                    _ => {}
                }
                if ring_size == 3 {
                    t = 60.0;
                } else if ring_size == 4 {
                    t = 90.0;
                }
                t
            }
        };
        let mut z = [0.0f64; 3];
        let mut c = [0.0f64; 3];
        for i in 0..3 {
            let (zi, ci) = match an[i] {
                1 => (1.395, 0.0),
                6 => (2.494, 1.016),
                7 => (2.711, 1.113),
                8 => (3.045, 1.337),
                9 => (2.847, 0.0),
                14 => (2.350, 0.811),
                15 => (2.350, 1.068),
                16 => (2.980, 1.249),
                17 => (2.909, 1.078),
                35 => (3.017, 0.0),
                53 => (3.086, 0.0),
                _ => (0.0, 0.0),
            };
            z[i] = zi;
            c[i] = ci;
        }
        let r0_ij = b1.r0;
        let r0_jk = b2.r0;
        let d = (r0_ij - r0_jk) * (r0_ij - r0_jk) / ((r0_ij + r0_jk) * (r0_ij + r0_jk));
        let theta0_rad = super::DEG2RAD * theta0;
        let mut beta = 1.75;
        if ring_size == 4 {
            beta *= 0.85;
        } else if ring_size == 3 {
            beta *= 0.05;
        }
        let ka = beta * z[0] * c[1] * z[2] / ((r0_ij + r0_jk) * theta0_rad * theta0_rad * (2.0 * d).exp());
        MmffAngle { ka, theta0 }
    }

    /// RDKit `getMMFFAngleBendParams`: `(angleType, params)`.
    pub fn angle_bend_params(&self, i1: usize, i2: usize, i3: usize) -> Option<(u8, MmffAngle)> {
        let mol = &self.tm.mol;
        if !self.valid || mol.bond_between(i1, i2).is_none() || mol.bond_between(i2, i3).is_none() {
            return None;
        }
        let p = self.params;
        let idx = [i1, i2, i3];
        let angle_type = self.angle_type(i1, i2, i3);
        let at = [self.atom_types[i1], self.atom_types[i2], self.atom_types[i3]];
        let mut params = p.angle.get(&p.def, angle_type, at[0], at[1], at[2]);
        let central = self.prop(at[1]);
        if params.is_none() || is_double_zero(params.unwrap().ka) {
            let b1 = self.bond_stretch_params(idx[0], idx[1]);
            let b2 = self.bond_stretch_params(idx[1], idx[2]);
            if let (Some((_, b1)), Some((_, b2))) = (b1, b2) {
                params = Some(self.angle_bend_empirical(params, central, &b1, &b2, idx));
            }
        }
        params.map(|x| (angle_type, x))
    }

    /// RDKit `getMMFFStretchBendParams`: `(stbnType, stbn, [bond1, bond2],
    /// angle)`.
    pub fn stretch_bend_params(&self, i1: usize, i2: usize, i3: usize) -> Option<(u8, MmffStbn, [MmffBond; 2], MmffAngle)> {
        if !self.valid {
            return None;
        }
        let p = self.params;
        let central = self.prop(self.atom_types[i2]);
        if central.linh != 0 {
            return None;
        }
        let idx = [i1, i2, i3];
        let at = [self.atom_types[i1], self.atom_types[i2], self.atom_types[i3]];
        let (bt1, b1) = self.bond_stretch_params(idx[0], idx[1])?;
        let (bt2, b2) = self.bond_stretch_params(idx[1], idx[2])?;
        let (angle_type, angle) = self.angle_bend_params(i1, i2, i3)?;
        let stbn_type = stretch_bend_type(
            angle_type,
            if at[0] <= at[2] { bt1 } else { bt2 },
            if at[0] < at[2] { bt2 } else { bt1 },
        );
        let mut r = p.stbn.get(stbn_type, bt1, bt2, at[0], at[1], at[2]);
        if r.1.is_none() {
            r = p.dfsb.get(
                periodic_table_row(self.tm.anum(i1)),
                periodic_table_row(self.tm.anum(i2)),
                periodic_table_row(self.tm.anum(i3)),
            );
        }
        let s = r.1?;
        if is_double_zero(s.kba_ijk) && is_double_zero(s.kba_kji) {
            return None;
        }
        let s = if r.0 { MmffStbn { kba_ijk: s.kba_kji, kba_kji: s.kba_ijk } } else { s };
        Some((stbn_type, s, [b1, b2], angle))
    }

    /// RDKit `getMMFFTorsionEmpiricalRuleParams`.
    fn torsion_empirical(&self, i2: usize, i3: usize) -> MmffTor {
        let jt = self.atom_types[i2];
        let kt = self.atom_types[i3];
        let jp = self.prop(jt);
        let kp = self.prop(kt);
        let mol = &self.tm.mol;
        let bond = mol.bond(mol.bond_between(i2, i3).expect("bond"));
        let mut u = [0.0f64; 2];
        let mut v = [0.0f64; 2];
        let mut w = [0.0f64; 2];
        let n_jk = ((jp.crd as i32 - 1) * (kp.crd as i32 - 1)) as f64;
        let an = [self.tm.anum(i2), self.tm.anum(i3)];
        for i in 0..2 {
            match an[i] {
                6 => {
                    u[i] = 2.0;
                    v[i] = 2.12;
                }
                7 => {
                    u[i] = 2.0;
                    v[i] = 1.5;
                }
                8 => {
                    u[i] = 2.0;
                    v[i] = 0.2;
                    w[i] = 2.0;
                }
                14 => {
                    u[i] = 1.25;
                    v[i] = 1.22;
                }
                15 => {
                    u[i] = 1.25;
                    v[i] = 2.40;
                }
                16 => {
                    u[i] = 1.25;
                    v[i] = 0.49;
                    w[i] = 8.0;
                }
                _ => {}
            }
        }
        let mut t = MmffTor::default();
        let row = periodic_table_row;
        if jp.linh != 0 || kp.linh != 0 {
            // rule (a)
        } else if is_mmff_aromatic(jt) && is_mmff_aromatic(kt) && bond.is_aromatic {
            // rule (b)
            let beta = if (jp.val == 3 && kp.val == 4) || (jp.val == 4 && kp.val == 3) { 3.0 } else { 6.0 };
            let pi_jk = if jp.pilp == 0 && kp.pilp == 0 { 0.5 } else { 0.3 };
            t.v2 = beta * pi_jk * (u[0] * u[1]).sqrt();
        } else if bond.order == BondOrder::Double {
            // rule (c)
            let beta = 6.0;
            let pi_jk = if jp.mltb == 2 && kp.mltb == 2 { 1.0 } else { 0.4 };
            t.v2 = beta * pi_jk * (u[0] * u[1]).sqrt();
        } else if jp.crd == 4 && kp.crd == 4 {
            // rule (d)
            t.v3 = (v[0] * v[1]).sqrt() / n_jk;
        } else if jp.crd == 4 && kp.crd != 4 {
            // rule (e)
            if (kp.crd == 3 && ((kp.val == 4 || kp.val == 34) || kp.mltb != 0)) || (kp.crd == 2 && (kp.val == 3 || kp.mltb != 0)) {
                // zero
            } else {
                t.v3 = (v[0] * v[1]).sqrt() / n_jk;
            }
        } else if kp.crd == 4 && jp.crd != 4 {
            // rule (f)
            if (jp.crd == 3 && ((jp.val == 4 || jp.val == 34) || jp.mltb != 0)) || (jp.crd == 2 && (jp.val == 3 || jp.mltb != 0)) {
                // zero
            } else {
                t.v3 = (v[0] * v[1]).sqrt() / n_jk;
            }
        } else if (bond.order == BondOrder::Single && jp.mltb != 0 && kp.mltb != 0)
            || (jp.mltb != 0 && kp.pilp != 0)
            || (jp.pilp != 0 && kp.mltb != 0)
        {
            // rule (g)
            if jp.pilp != 0 && kp.pilp != 0 {
                // case (1): zero
            } else if jp.pilp != 0 && kp.mltb != 0 {
                let beta = 6.0;
                let mut pi_jk = 0.0;
                if jp.mltb == 1 {
                    pi_jk = 0.5;
                } else if row(an[0]) == 2 && row(an[1]) == 2 {
                    pi_jk = 0.3;
                } else if row(an[0]) != 2 || row(an[1]) != 2 {
                    pi_jk = 0.15;
                }
                t.v2 = beta * pi_jk * (u[0] * u[1]).sqrt();
            } else if kp.pilp != 0 && jp.mltb != 0 {
                let beta = 6.0;
                let mut pi_jk = 0.0;
                if kp.mltb == 1 {
                    pi_jk = 0.5;
                } else if row(an[0]) == 2 && row(an[1]) == 2 {
                    pi_jk = 0.3;
                } else if row(an[0]) != 2 || row(an[1]) != 2 {
                    pi_jk = 0.15;
                }
                t.v2 = beta * pi_jk * (u[0] * u[1]).sqrt();
            } else if (jp.mltb == 1 || kp.mltb == 1) && (an[0] != 6 || an[1] != 6) {
                t.v2 = 6.0 * 0.4 * (u[0] * u[1]).sqrt();
            } else {
                t.v2 = 6.0 * 0.15 * (u[0] * u[1]).sqrt();
            }
        } else {
            // rule (h)
            if (an[0] == 8 || an[0] == 16) && (an[1] == 8 || an[1] == 16) {
                t.v2 = -(w[0] * w[1]).sqrt();
            } else {
                t.v3 = (v[0] * v[1]).sqrt() / n_jk;
            }
        }
        t
    }

    /// RDKit `getMMFFTorsionParams`: `(torsionType, params)`; `None` if the
    /// torsion has no (non-zero) parameters.
    pub fn torsion_params(&self, i1: usize, i2: usize, i3: usize, i4: usize) -> Option<(u8, MmffTor)> {
        let mol = &self.tm.mol;
        if !self.valid || mol.bond_between(i1, i2).is_none() || mol.bond_between(i2, i3).is_none() || mol.bond_between(i3, i4).is_none() {
            return None;
        }
        let p = self.params;
        let at = [self.atom_types[i1], self.atom_types[i2], self.atom_types[i3], self.atom_types[i4]];
        let tt = self.torsion_type(i1, i2, i3, i4);
        let (used, params) = p.tor.get(&p.def, tt, at[0], at[1], at[2], at[3]);
        let mut tor_type = if used != 0 { used } else { tt.0 };
        let params = match params {
            Some(x) => x,
            None => {
                tor_type = tt.0;
                self.torsion_empirical(i2, i3)
            }
        };
        if is_double_zero(params.v1) && is_double_zero(params.v2) && is_double_zero(params.v3) {
            return None;
        }
        Some((tor_type, params))
    }

    /// RDKit `getMMFFOopBendParams`.
    pub fn oop_bend_params(&self, i1: usize, i2: usize, i3: usize, i4: usize) -> Option<MmffOop> {
        let mol = &self.tm.mol;
        if !self.valid || mol.bond_between(i1, i2).is_none() || mol.bond_between(i2, i3).is_none() || mol.bond_between(i2, i4).is_none() {
            return None;
        }
        let p = self.params;
        p.oop.get(&p.def, self.atom_types[i1], self.atom_types[i2], self.atom_types[i3], self.atom_types[i4])
    }

    /// RDKit `getMMFFVdWParams`.
    pub fn vdw_params(&self, i: usize, j: usize) -> Option<MmffVdwRijstarEps> {
        if !self.valid {
            return None;
        }
        let tab = &self.params.vdw;
        let pi = tab.get(self.atom_types[i])?;
        let pj = tab.get(self.atom_types[j])?;
        let r_unscaled = super::terms::calc_unscaled_vdw_minimum(tab, pi, pj);
        let eps_unscaled = super::terms::calc_unscaled_vdw_well_depth(r_unscaled, pi, pj);
        let mut r = MmffVdwRijstarEps {
            r_ij_star_unscaled: r_unscaled,
            epsilon_unscaled: eps_unscaled,
            r_ij_star: r_unscaled,
            epsilon: eps_unscaled,
        };
        super::terms::scale_vdw_params(&mut r.r_ij_star, &mut r.epsilon, tab, pi, pj);
        Some(r)
    }
}

/// RDKit `getMMFFStretchBendType`.
pub(crate) fn stretch_bend_type(angle_type: u8, bt1: u8, bt2: u8) -> u8 {
    match angle_type {
        1 => {
            if bt1 != 0 || bt1 == bt2 { 1 } else { 2 }
        }
        2 => 3,
        4 => 4,
        3 => 5,
        5 => {
            if bt1 != 0 || bt1 == bt2 { 6 } else { 7 }
        }
        6 => 8,
        7 => {
            if bt1 != 0 || bt1 == bt2 { 9 } else { 10 }
        }
        8 => 11,
        _ => 0,
    }
}
