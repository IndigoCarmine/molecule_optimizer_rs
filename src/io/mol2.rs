//! Tripos MOL2 reading/writing.
//!
//! The line-preserving record model ([`Mol2File`], [`Mol2Line`],
//! [`Mol2AtomRecord`], [`Mol2BondRecord`]) is adapted from
//! `manul_cat_rs/src/parsing/mol2.rs` by the same author (Yuhei Yamada,
//! IndigoCarmine), changed to f64 Å coordinates and extended with
//! conversion to/from [`Molecule`], multi-molecule files, CRYSIN cells and
//! SYBYL atom typing for output.

use crate::core::cell::UnitCell;
use crate::core::{Atom, BondOrder, Hybridization, Molecule, atomic_number};
use crate::error::{Error, Result};
use crate::perception;
use std::collections::HashMap;
use std::fmt::Write;

// --- Mol2 records ---

#[derive(Debug, Clone, PartialEq)]
pub struct Mol2AtomRecord {
    pub atom_id: usize,
    pub atom_name: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub atom_type: String,
    pub subst_id: i32,
    pub subst_name: String,
    pub charge: f64,
    pub status_bit: String,
}

impl Mol2AtomRecord {
    /// Parses an ATOM line. Columns after `atom_type` are optional in the
    /// format; missing ones default to `1 / UNL / 0.0`.
    pub fn from_line(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 6 {
            return None;
        }
        Some(Mol2AtomRecord {
            atom_id: parts[0].parse().ok()?,
            atom_name: parts[1].to_string(),
            x: parts[2].parse().ok()?,
            y: parts[3].parse().ok()?,
            z: parts[4].parse().ok()?,
            atom_type: parts[5].to_string(),
            subst_id: parts.get(6).and_then(|s| s.parse().ok()).unwrap_or(1),
            subst_name: parts.get(7).unwrap_or(&"UNL").to_string(),
            charge: parts.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            status_bit: parts.get(9).unwrap_or(&"").to_string(),
        })
    }

    pub fn to_line(&self) -> String {
        let mut s = format!(
            "{:>7} {:<8}{:>11.4}{:>11.4}{:>11.4} {:<6}{:>5} {:<8}{:>10.4}",
            self.atom_id, self.atom_name, self.x, self.y, self.z, self.atom_type, self.subst_id, self.subst_name, self.charge
        );
        if !self.status_bit.is_empty() {
            s.push(' ');
            s.push_str(&self.status_bit);
        }
        s
    }

    /// Element symbol from the SYBYL type (`"C.ar"` → `"C"`, `"Cl"` → `"Cl"`).
    pub fn element_symbol(&self) -> &str {
        self.atom_type.split('.').next().unwrap_or("")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mol2BondRecord {
    pub bond_id: usize,
    pub origin_atom_id: usize,
    pub target_atom_id: usize,
    pub bond_type: String,
    pub status_bit: String,
}

impl Mol2BondRecord {
    pub fn from_line(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return None;
        }
        Some(Mol2BondRecord {
            bond_id: parts[0].parse().ok()?,
            origin_atom_id: parts[1].parse().ok()?,
            target_atom_id: parts[2].parse().ok()?,
            bond_type: parts[3].to_string(),
            status_bit: parts.get(4).unwrap_or(&"").to_string(),
        })
    }

    pub fn to_line(&self) -> String {
        let mut s = format!("{:>6}{:>6}{:>6} {:<4}", self.bond_id, self.origin_atom_id, self.target_atom_id, self.bond_type);
        if !self.status_bit.is_empty() {
            s.push(' ');
            s.push_str(&self.status_bit);
        }
        s.trim_end().to_string()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Mol2Line {
    Atom(Mol2AtomRecord),
    Bond(Mol2BondRecord),
    SectionHeader(String),
    MoleculeLine(String),
    Other(String),
    Empty,
}

/// A MOL2 file as a list of lines; [`Mol2File::dump`] reproduces the
/// sections in order (atom/bond lines are re-formatted).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mol2File {
    pub lines: Vec<Mol2Line>,
}

impl Mol2File {
    pub fn load(content: &str) -> Self {
        let mut lines = Vec::new();
        let mut current_section = String::new();
        for line in content.lines() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() {
                lines.push(Mol2Line::Empty);
                continue;
            }
            if let Some(section) = line.trim_start().strip_prefix("@<TRIPOS>") {
                current_section = section.trim().to_ascii_uppercase();
                lines.push(Mol2Line::SectionHeader(line.to_string()));
                continue;
            }
            match current_section.as_str() {
                "MOLECULE" => lines.push(Mol2Line::MoleculeLine(line.to_string())),
                "ATOM" => match Mol2AtomRecord::from_line(line) {
                    Some(a) => lines.push(Mol2Line::Atom(a)),
                    None => lines.push(Mol2Line::Other(line.to_string())),
                },
                "BOND" => match Mol2BondRecord::from_line(line) {
                    Some(b) => lines.push(Mol2Line::Bond(b)),
                    None => lines.push(Mol2Line::Other(line.to_string())),
                },
                _ => lines.push(Mol2Line::Other(line.to_string())),
            }
        }
        Self { lines }
    }

    pub fn dump(&self) -> String {
        let mut out = String::new();
        for line in &self.lines {
            match line {
                Mol2Line::Atom(a) => writeln!(out, "{}", a.to_line()).unwrap(),
                Mol2Line::Bond(b) => writeln!(out, "{}", b.to_line()).unwrap(),
                Mol2Line::SectionHeader(s) | Mol2Line::MoleculeLine(s) | Mol2Line::Other(s) => writeln!(out, "{}", s).unwrap(),
                Mol2Line::Empty => writeln!(out).unwrap(),
            }
        }
        out
    }

    pub fn atoms(&self) -> impl Iterator<Item = &Mol2AtomRecord> {
        self.lines.iter().filter_map(|l| match l {
            Mol2Line::Atom(a) => Some(a),
            _ => None,
        })
    }

    pub fn bonds(&self) -> impl Iterator<Item = &Mol2BondRecord> {
        self.lines.iter().filter_map(|l| match l {
            Mol2Line::Bond(b) => Some(b),
            _ => None,
        })
    }

    /// Splits a multi-molecule file into one [`Mol2File`] per
    /// `@<TRIPOS>MOLECULE` block.
    pub fn split_molecules(&self) -> Vec<Mol2File> {
        let mut out: Vec<Mol2File> = Vec::new();
        for l in &self.lines {
            if let Mol2Line::SectionHeader(h) = l {
                if h.trim().eq_ignore_ascii_case("@<TRIPOS>MOLECULE") {
                    out.push(Mol2File::default());
                }
            }
            match out.last_mut() {
                Some(m) => m.lines.push(l.clone()),
                None => {
                    if !matches!(l, Mol2Line::Empty | Mol2Line::Other(_)) {
                        out.push(Mol2File { lines: vec![l.clone()] });
                    }
                }
            }
        }
        out
    }

    /// Lines of a named section (excluding the header).
    pub fn section_lines(&self, name: &str) -> Vec<&str> {
        let mut inside = false;
        let mut out = Vec::new();
        for l in &self.lines {
            match l {
                Mol2Line::SectionHeader(h) => {
                    inside = h.trim().trim_start_matches("@<TRIPOS>").eq_ignore_ascii_case(name);
                }
                Mol2Line::MoleculeLine(s) | Mol2Line::Other(s) if inside => out.push(s.as_str()),
                _ => {}
            }
        }
        out
    }

    /// Partial charges of the ATOM records (in file order).
    pub fn partial_charges(&self) -> Vec<f64> {
        self.atoms().filter(|a| !is_lone_pair(a)).map(|a| a.charge).collect()
    }

    /// Converts the first molecule block to a [`Molecule`].
    pub fn to_molecule(&self, opts: &Mol2ReadOptions) -> Result<Molecule> {
        mol2_to_molecule(self, opts)
    }
}

fn is_lone_pair(a: &Mol2AtomRecord) -> bool {
    let t = a.atom_type.to_ascii_uppercase();
    t == "LP" || t == "LP."
}

/// Options for reading MOL2.
#[derive(Debug, Clone)]
pub struct Mol2ReadOptions {
    /// Treat every hydrogen as present in the file (no implicit H). MOL2
    /// files almost always contain all hydrogens. Default `true`.
    pub all_hydrogens_explicit: bool,
    /// Assign formal charges from SYBYL types / valences (MOL2 has no formal
    /// charge field): carboxylate/nitro `O.co2`, `N.4`, guanidinium/amidinium
    /// `C.cat`, phosphate/sulfonate oxygens. Default `true`.
    pub infer_formal_charges: bool,
    /// Run [`perception::sanitize`]. Default `true`.
    pub sanitize: bool,
}

impl Default for Mol2ReadOptions {
    fn default() -> Self {
        Mol2ReadOptions { all_hydrogens_explicit: true, infer_formal_charges: true, sanitize: true }
    }
}

fn mol2_bond_order(t: &str) -> BondOrder {
    match t.to_ascii_lowercase().as_str() {
        "1" | "am" => BondOrder::Single,
        "2" => BondOrder::Double,
        "3" => BondOrder::Triple,
        "ar" => BondOrder::Aromatic,
        "du" | "un" | "nc" => BondOrder::Zero,
        _ => BondOrder::Single,
    }
}

fn mol2_to_molecule(file: &Mol2File, opts: &Mol2ReadOptions) -> Result<Molecule> {
    let mut mol = Molecule::new();
    if let Some(name) = file.section_lines("MOLECULE").first() {
        mol.name = name.trim().to_string();
    }
    let mut id_to_index: HashMap<usize, usize> = HashMap::new();
    let mut types: Vec<String> = Vec::new();
    for rec in file.atoms() {
        if is_lone_pair(rec) {
            continue;
        }
        let sym = rec.element_symbol();
        let anum = if sym.eq_ignore_ascii_case("Du") || sym == "*" {
            0
        } else {
            atomic_number(sym)
                .or_else(|| atomic_number(&rec.atom_name.chars().take_while(|c| c.is_ascii_alphabetic()).collect::<String>()))
                .ok_or_else(|| Error::UnknownElement(rec.atom_type.clone()))?
        };
        let mut atom = Atom::new(anum);
        atom.no_implicit = opts.all_hydrogens_explicit;
        atom.residue = Some(crate::core::ResidueInfo {
            atom_name: rec.atom_name.clone(),
            residue_name: rec.subst_name.clone(),
            residue_number: rec.subst_id,
            serial: rec.atom_id as i32,
            occupancy: 1.0,
            ..Default::default()
        });
        let idx = mol.add_atom(atom, [rec.x, rec.y, rec.z]);
        id_to_index.insert(rec.atom_id, idx);
        types.push(rec.atom_type.clone());
    }
    for rec in file.bonds() {
        let (Some(&a), Some(&b)) = (id_to_index.get(&rec.origin_atom_id), id_to_index.get(&rec.target_atom_id)) else {
            continue; // bonds to lone pairs / unknown atoms
        };
        if a != b && !rec.bond_type.eq_ignore_ascii_case("nc") {
            let order = mol2_bond_order(&rec.bond_type);
            mol.add_bond(a, b, order);
            if order == BondOrder::Aromatic {
                mol.atom_mut(a).is_aromatic = true;
                mol.atom_mut(b).is_aromatic = true;
            }
        }
    }
    // CRYSIN: a b c alpha beta gamma [space group] [setting]
    if let Some(line) = file.section_lines("CRYSIN").first() {
        let v: Vec<f64> = line.split_whitespace().take(6).filter_map(|s| s.parse().ok()).collect();
        if v.len() == 6 && v[0] > 0.0 && v[1] > 0.0 && v[2] > 0.0 {
            mol.cell = UnitCell::from_parameters(v[0], v[1], v[2], v[3], v[4], v[5]).ok();
        }
    }
    if opts.infer_formal_charges && !read_unity_charges(file, &mut mol, &id_to_index) {
        cleanup_substructures(&mut mol, &types)?;
        guess_formal_charges(&mut mol, &types);
    }
    if opts.sanitize {
        perception::sanitize(&mut mol)?;
    }
    Ok(mol)
}

/// Formal charges from a `@<TRIPOS>UNITY_ATOM_ATTR` section (`AtomExpr <q>`
/// entries, as read by RDKit). Returns `true` if any charge was found.
fn read_unity_charges(file: &Mol2File, mol: &mut Molecule, id_to_index: &HashMap<usize, usize>) -> bool {
    let lines = file.section_lines("UNITY_ATOM_ATTR");
    let mut found = false;
    let mut k = 0;
    while k < lines.len() {
        let f: Vec<&str> = lines[k].split_whitespace().collect();
        k += 1;
        let (Some(id), Some(n)) = (f.first().and_then(|s| s.parse::<usize>().ok()), f.get(1).and_then(|s| s.parse::<usize>().ok())) else {
            continue;
        };
        for _ in 0..n {
            if k >= lines.len() {
                break;
            }
            let g: Vec<&str> = lines[k].split_whitespace().collect();
            k += 1;
            if g.first() == Some(&"AtomExpr") {
                if let Some(q) = g.get(1).filter(|s| !s.contains('=')).and_then(|s| s.parse::<i8>().ok()) {
                    if let Some(&i) = id_to_index.get(&id) {
                        mol.atom_mut(i).formal_charge = q;
                        found = true;
                    }
                }
            }
        }
    }
    found
}

fn set_bond(mol: &mut Molecule, a: usize, b: usize, order: BondOrder) {
    if let Some(bi) = mol.bond_between(a, b) {
        let bd = mol.bond_mut(bi);
        bd.order = order;
        bd.is_aromatic = false;
    }
}

fn no_h_neighbors_n_oxide(mol: &Molecule, n: usize, to_mod: &mut Option<usize>) -> usize {
    let mut nh = 0;
    for x in mol.neighbors(n) {
        if mol.atom(x).atomic_num == 1 {
            nh += 1;
        } else if mol.atom(x).atomic_num == 8 && mol.degree(x) == 1 {
            *to_mod = Some(n);
        }
    }
    nh
}

/// Port of RDKit `cleanUpMol2Substructures` (`Mol2FileParser.cpp`,
/// BSD-3-Clause, Copyright (C) 2008 Novartis and other RDKit contributors):
/// resolves `N.4`, `O.co2` (carboxylate / phosphate / sulfonate) and `C.cat`
/// (guanidinium / amidinium) groups into one Kekulé form with charges,
/// choosing the same resonance form RDKit does.
fn cleanup_substructures(mol: &mut Molecule, types: &[String]) -> Result<()> {
    let n = mol.num_atoms();
    let mut fixed = vec![false; n];
    for idx in 0..n {
        if fixed[idx] {
            continue;
        }
        let t = types[idx].as_str();
        if t == "N.4" {
            mol.atom_mut(idx).formal_charge = 1;
        } else if t == "O.co2" {
            if mol.degree(idx) != 1 {
                return Err(Error::Perception(format!("{}: O.co2 with degree > 1", mol.name)));
            }
            let nbr = mol.neighbors(idx).next().unwrap();
            let tn = types[nbr].as_str();
            if tn == "P.3" {
                set_bond(mol, idx, nbr, BondOrder::Double);
                mol.atom_mut(idx).is_aromatic = false;
                fixed[idx] = true;
                let others: Vec<usize> = mol.neighbors(nbr).collect();
                for o in others {
                    if mol.atom(o).atomic_num == 8 && !fixed[o] && types[o] == "O.co2" {
                        set_bond(mol, nbr, o, BondOrder::Single);
                        mol.atom_mut(o).formal_charge = -1;
                        mol.atom_mut(o).is_aromatic = false;
                        fixed[o] = true;
                    }
                }
                mol.atom_mut(nbr).is_aromatic = false;
                fixed[nbr] = true;
            } else if tn == "C.2" || tn == "S.o2" {
                if !fixed[nbr] {
                    set_bond(mol, idx, nbr, BondOrder::Single);
                    mol.atom_mut(idx).formal_charge = -1;
                    mol.atom_mut(idx).is_aromatic = false;
                    mol.atom_mut(nbr).is_aromatic = false;
                    fixed[idx] = true;
                    fixed[nbr] = true;
                } else {
                    set_bond(mol, idx, nbr, BondOrder::Double);
                    mol.atom_mut(idx).is_aromatic = false;
                    fixed[idx] = true;
                }
            } else {
                return Err(Error::Perception(format!("{}: O.co2 with non C.2 or S.o2 neighbor", mol.name)));
            }
        } else if t == "C.cat" {
            fixed[idx] = true;
            let nbrs: Vec<usize> = mol.neighbors(idx).collect();
            let n_n = nbrs.iter().filter(|&&x| mol.atom(x).atomic_num == 7).count();
            if !(2..=3).contains(&n_n) {
                return Err(Error::Perception(format!("{}: C.cat with bad number of N neighbors", mol.name)));
            }
            let mut to_mod: Option<usize> = None;
            if n_n == 2 {
                let mut first: Option<usize> = None;
                let mut second: Option<usize> = None;
                for &x in &nbrs {
                    if mol.atom(x).atomic_num == 7 {
                        set_bond(mol, idx, x, BondOrder::Single);
                        mol.atom_mut(x).is_aromatic = false;
                        fixed[x] = true;
                        if first.is_some() {
                            second = Some(x);
                        } else {
                            first = Some(x);
                        }
                    }
                }
                let i1 = first.unwrap();
                let i2 = second.unwrap_or(i1);
                let nh1 = no_h_neighbors_n_oxide(mol, i1, &mut to_mod);
                let nh2 = no_h_neighbors_n_oxide(mol, i2, &mut to_mod);
                if to_mod.is_none() {
                    to_mod = Some(if nh1 != nh2 {
                        if nh1 > nh2 { i1 } else { i2 }
                    } else {
                        let ri = mol.rings().clone();
                        if ri.num_atom_rings(i1) > ri.num_atom_rings(i2) { i1 } else { i2 }
                    });
                }
            } else {
                let mut lowest = 100usize;
                for &x in &nbrs {
                    if !fixed[x] {
                        let mut deg = 0;
                        for y in mol.neighbors(x) {
                            if mol.atom(y).atomic_num > 1 {
                                deg += if types[y] == "C.cat" { 2 } else { 1 };
                            }
                        }
                        if deg < lowest {
                            to_mod = Some(x);
                            lowest = deg;
                        }
                        set_bond(mol, idx, x, BondOrder::Single);
                        mol.atom_mut(x).is_aromatic = false;
                        fixed[x] = true;
                    } else {
                        set_bond(mol, idx, x, BondOrder::Single);
                    }
                }
            }
            if let Some(m) = to_mod {
                set_bond(mol, idx, m, BondOrder::Double);
                mol.atom_mut(m).formal_charge = 1;
            }
            mol.atom_mut(idx).is_aromatic = false;
        }
    }
    Ok(())
}

/// Port of RDKit `guessFormalCharges` (`Mol2FileParser.cpp`): assigns
/// formal charges to uncharged non-carbon atoms from their bond valence.
fn guess_formal_charges(mol: &mut Molecule, types: &[String]) {
    mol.ring_info = None;
    let ri = mol.rings().clone();
    for i in 0..mol.num_atoms() {
        let a = mol.atom(i).clone();
        if a.formal_charge != 0 || a.atomic_num == 6 || a.atomic_num == 0 {
            continue;
        }
        let mut n_arom = 0;
        let mut accum = 0.0;
        for &(_, b) in mol.atom_bonds(i) {
            accum += mol.bond(b).valence_contrib(i);
            if mol.bond(b).order == BondOrder::Aromatic {
                n_arom += 1;
            }
        }
        let t = types[i].as_str();
        if a.is_aromatic && !t.contains("ar") && ri.is_atom_in_ring_of_size(i, 5) {
            continue;
        }
        if n_arom == 3 && t == "N.ar" {
            continue;
        }
        let exp_val = (accum + 0.1).round() as i32;
        let el = crate::core::element(a.atomic_num);
        let Some(&v0) = el.valences.first() else { continue };
        let n_el = el.n_outer as i32;
        let mut chg = if n_el >= 4 { exp_val - v0 } else { v0 - exp_val };
        if chg > 0 && n_el >= 4 {
            for &vi in &el.valences {
                chg = exp_val - vi;
                if vi <= exp_val && chg < 2 {
                    break;
                }
            }
        }
        if chg != 0 {
            let q = if a.is_aromatic && chg.abs() > 1 { chg.signum() } else { chg };
            mol.atom_mut(i).formal_charge = q as i8;
            if chg == 2 && exp_val == 5 && a.atomic_num == 7 {
                fix_nitro(mol, i);
            }
        }
    }
}

/// N(=O)(=O) written as a pentavalent nitrogen → N+(=O)O-.
fn fix_nitro(mol: &mut Molecule, n: usize) {
    let os: Vec<(usize, usize)> = mol
        .atom_bonds(n)
        .iter()
        .copied()
        .filter(|&(o, b)| mol.atom(o).atomic_num == 8 && mol.bond(b).order == BondOrder::Double)
        .collect();
    if let Some(&(o, b)) = os.get(1) {
        mol.bond_mut(b).order = BondOrder::Single;
        mol.atom_mut(o).formal_charge = -1;
        mol.atom_mut(n).formal_charge = 1;
    }
}

/// Reads all molecules of a MOL2 file.
pub fn read_mol2_str(text: &str) -> Result<Vec<Molecule>> {
    read_mol2_with(text, &Mol2ReadOptions::default())
}

pub fn read_mol2_with(text: &str, opts: &Mol2ReadOptions) -> Result<Vec<Molecule>> {
    let file = Mol2File::load(text);
    let blocks = file.split_molecules();
    if blocks.is_empty() {
        return Err(Error::parse(1, "no @<TRIPOS>MOLECULE block"));
    }
    blocks.iter().map(|b| b.to_molecule(opts)).collect()
}

/// Reads the first molecule of a MOL2 file.
pub fn read_mol2(text: &str) -> Result<Molecule> {
    Ok(read_mol2_str(text)?.remove(0))
}

// --- writing ---

fn has_double_to(mol: &Molecule, i: usize, elems: &[u8]) -> bool {
    mol.atom_bonds(i).iter().any(|&(o, b)| mol.bond(b).order == BondOrder::Double && elems.contains(&mol.atom(o).atomic_num))
}

fn terminal_o_count(mol: &Molecule, i: usize) -> usize {
    mol.neighbors(i).filter(|&o| mol.atom(o).atomic_num == 8 && mol.degree(o) == 1).count()
}

/// SYBYL atom type for atom `i` (expects a sanitized molecule).
pub fn sybyl_type(mol: &Molecule, i: usize) -> String {
    let a = mol.atom(i);
    let deg = mol.degree(i) + a.total_hidden_hs() as usize;
    match a.atomic_num {
        0 => "Du".into(),
        1 => "H".into(),
        6 => {
            if a.is_aromatic {
                "C.ar".into()
            } else if a.hybridization == Hybridization::SP {
                "C.1".into()
            } else if a.hybridization == Hybridization::SP2 {
                let n_n = mol.neighbors(i).filter(|&x| mol.atom(x).atomic_num == 7).count();
                let cationic = n_n == 3 && mol.neighbors(i).any(|x| mol.atom(x).formal_charge > 0);
                if a.formal_charge > 0 || cationic { "C.cat".into() } else { "C.2".into() }
            } else {
                "C.3".into()
            }
        }
        7 => {
            if a.is_aromatic {
                if deg == 3 && mol.atom_bonds(i).iter().all(|&(_, b)| mol.bond(b).order != BondOrder::Double) {
                    "N.pl3".into()
                } else {
                    "N.ar".into()
                }
            } else if a.hybridization == Hybridization::SP {
                "N.1".into()
            } else if deg == 4 && a.formal_charge > 0 && mol.atom_bonds(i).iter().all(|&(_, b)| mol.bond(b).order == BondOrder::Single) {
                "N.4".into()
            } else if terminal_o_count(mol, i) >= 2 {
                "N.pl3".into()
            } else if mol.neighbors(i).any(|c| mol.atom(c).atomic_num == 6 && has_double_to(mol, c, &[8, 16]) && deg == 3) {
                "N.am".into()
            } else if mol.atom_bonds(i).iter().any(|&(_, b)| mol.bond(b).order == BondOrder::Double) {
                "N.2".into()
            } else if a.hybridization == Hybridization::SP2 {
                "N.pl3".into()
            } else {
                "N.3".into()
            }
        }
        8 => {
            let c = mol.neighbors(i).next();
            let co2 = mol.degree(i) == 1
                && c.is_some_and(|c| {
                    let z = mol.atom(c).atomic_num;
                    (z == 6 || z == 15) && terminal_o_count(mol, c) >= 2 && (a.formal_charge < 0 || mol.neighbors(c).any(|o| mol.atom(o).formal_charge < 0))
                });
            if co2 {
                "O.co2".into()
            } else if mol.atom_bonds(i).iter().any(|&(_, b)| mol.bond(b).order == BondOrder::Double) {
                "O.2".into()
            } else {
                "O.3".into()
            }
        }
        15 => "P.3".into(),
        16 => {
            let n_oxo = mol.atom_bonds(i).iter().filter(|&&(o, b)| mol.atom(o).atomic_num == 8 && mol.bond(b).order == BondOrder::Double && mol.degree(o) == 1).count();
            if n_oxo >= 2 {
                "S.O2".into()
            } else if n_oxo == 1 {
                "S.O".into()
            } else if a.is_aromatic || mol.atom_bonds(i).iter().any(|&(_, b)| mol.bond(b).order == BondOrder::Double) {
                "S.2".into()
            } else {
                "S.3".into()
            }
        }
        _ => a.symbol().to_string(),
    }
}

fn mol2_bond_type(mol: &Molecule, kek: &Molecule, b: usize, aromatic_bonds: bool) -> &'static str {
    let bond = mol.bond(b);
    let aromatic = bond.is_aromatic || bond.order == BondOrder::Aromatic;
    if aromatic && aromatic_bonds {
        return "ar";
    }
    // Kekulé order (aromatic bonds resolved), which every MOL2 reader accepts
    match kek.bond(b).order {
        BondOrder::Double => "2",
        BondOrder::Triple => "3",
        BondOrder::Zero => "nc",
        BondOrder::Aromatic => "ar",
        _ => {
            if aromatic {
                return "1";
            }
            // amide C(=O)-N
            let (x, y) = (bond.begin, bond.end);
            let is_amide = |c: usize, nn: usize| {
                kek.atom(c).atomic_num == 6 && kek.atom(nn).atomic_num == 7 && has_double_to(kek, c, &[8])
            };
            if is_amide(x, y) || is_amide(y, x) { "am" } else { "1" }
        }
    }
}

/// Options for writing MOL2.
#[derive(Debug, Clone, Default)]
pub struct Mol2WriteOptions {
    /// Write aromatic bonds as `ar` instead of Kekulé orders. Kekulé orders
    /// (the default) are read correctly by all common readers, including
    /// RDKit, also for rings with exocyclic double bonds (pyridones, ...).
    pub aromatic_bonds: bool,
}

/// Builds a [`Mol2File`] for a molecule (optional replacement coordinates
/// and partial charges).
pub fn molecule_to_mol2(mol: &Molecule, coords: Option<&[[f64; 3]]>, charges: Option<&[f64]>) -> Mol2File {
    molecule_to_mol2_with(mol, coords, charges, &Mol2WriteOptions::default())
}

/// [`molecule_to_mol2`] with options.
pub fn molecule_to_mol2_with(
    mol: &Molecule,
    coords: Option<&[[f64; 3]]>,
    charges: Option<&[f64]>,
    opts: &Mol2WriteOptions,
) -> Mol2File {
    let c = coords.unwrap_or(&mol.coords);
    let kek = perception::kekulized_copy(mol).unwrap_or_else(|_| mol.clone());
    let mut lines = vec![
        Mol2Line::SectionHeader("@<TRIPOS>MOLECULE".into()),
        Mol2Line::MoleculeLine(if mol.name.is_empty() { "molecule".into() } else { mol.name.clone() }),
        Mol2Line::MoleculeLine(format!("{} {} 1 0 0", mol.num_atoms(), mol.num_bonds())),
        Mol2Line::MoleculeLine("SMALL".into()),
        Mol2Line::MoleculeLine(if charges.is_some() { "USER_CHARGES".into() } else { "NO_CHARGES".into() }),
        Mol2Line::Empty,
        Mol2Line::SectionHeader("@<TRIPOS>ATOM".into()),
    ];
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for i in 0..mol.num_atoms() {
        let a = mol.atom(i);
        let (name, subst_id, subst_name) = match &a.residue {
            Some(r) if !r.atom_name.is_empty() => (
                r.atom_name.clone(),
                if r.residue_number == 0 { 1 } else { r.residue_number },
                if r.residue_name.is_empty() { "UNL".into() } else { r.residue_name.clone() },
            ),
            _ => {
                let k = counts.entry(a.symbol()).or_insert(0);
                *k += 1;
                (format!("{}{}", a.symbol(), k), 1, "UNL".to_string())
            }
        };
        lines.push(Mol2Line::Atom(Mol2AtomRecord {
            atom_id: i + 1,
            atom_name: name,
            x: c[i][0],
            y: c[i][1],
            z: c[i][2],
            atom_type: sybyl_type(mol, i),
            subst_id,
            subst_name,
            charge: charges.map(|q| q[i]).unwrap_or(0.0),
            status_bit: String::new(),
        }));
    }
    lines.push(Mol2Line::SectionHeader("@<TRIPOS>BOND".into()));
    for b in 0..mol.num_bonds() {
        lines.push(Mol2Line::Bond(Mol2BondRecord {
            bond_id: b + 1,
            origin_atom_id: mol.bond(b).begin + 1,
            target_atom_id: mol.bond(b).end + 1,
            bond_type: mol2_bond_type(mol, &kek, b, opts.aromatic_bonds).to_string(),
            status_bit: String::new(),
        }));
    }
    if let Some(cell) = &mol.cell {
        let [a, b, cc] = cell.lengths();
        let [al, be, ga] = cell.angles();
        lines.push(Mol2Line::SectionHeader("@<TRIPOS>CRYSIN".into()));
        lines.push(Mol2Line::Other(format!("{a:.4} {b:.4} {cc:.4} {al:.4} {be:.4} {ga:.4} 1 1")));
    }
    Mol2File { lines }
}

/// Writes a molecule as MOL2 text.
pub fn write_mol2(mol: &Molecule) -> String {
    molecule_to_mol2(mol, None, None).dump()
}

pub fn write_mol2_coords(mol: &Molecule, coords: &[[f64; 3]]) -> String {
    molecule_to_mol2(mol, Some(coords), None).dump()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACETATE: &str = "@<TRIPOS>MOLECULE
acetate
 7 6 1 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3     1  ACT1       -0.1000
      2 C2          1.5200    0.0000    0.0000 C.2     1  ACT1        0.1000
      3 O1          2.1500    1.0800    0.0000 O.co2   1  ACT1       -0.6000
      4 O2          2.1500   -1.0800    0.0000 O.co2   1  ACT1       -0.6000
      5 H1         -0.3600    1.0300    0.0000 H       1  ACT1        0.0300
      6 H2         -0.3600   -0.5100    0.8900 H       1  ACT1        0.0300
      7 H3         -0.3600   -0.5100   -0.8900 H       1  ACT1        0.0300
@<TRIPOS>BOND
     1     1     2 1
     2     2     3 ar
     3     2     4 ar
     4     1     5 1
     5     1     6 1
     6     1     7 1
";

    #[test]
    fn carboxylate_charges_and_roundtrip() {
        let m = read_mol2(ACETATE).unwrap();
        assert_eq!(m.num_atoms(), 7);
        let q: i32 = m.atoms().iter().map(|a| a.formal_charge as i32).sum();
        assert_eq!(q, -1);
        assert!(!m.has_hidden_hydrogens());
        let f = Mol2File::load(ACETATE);
        assert_eq!(f.partial_charges().len(), 7);
        // write → read gives the same graph
        let text = write_mol2(&m);
        let m2 = read_mol2(&text).unwrap();
        assert_eq!(m2.num_bonds(), m.num_bonds());
        let q2: i32 = m2.atoms().iter().map(|a| a.formal_charge as i32).sum();
        assert_eq!(q2, -1);
        assert_eq!(Mol2File::load(&text).dump(), text);
    }

    #[test]
    fn multi_molecule_file() {
        let two = format!("{ACETATE}\n{ACETATE}");
        assert_eq!(read_mol2_str(&two).unwrap().len(), 2);
    }
}
