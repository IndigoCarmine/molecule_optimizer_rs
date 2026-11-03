//! Periodic table data.
//!
//! The table (`data/elements.tsv`) is taken from RDKit
//! `Code/GraphMol/atomic_data.cpp` (BSD-3-Clause, Copyright (c) 2006-2015,
//! Rational Discovery LLC, Greg Landrum, and others).

use std::sync::OnceLock;

/// Data for one element.
#[derive(Debug, Clone)]
pub struct ElementData {
    pub atomic_num: u8,
    pub symbol: &'static str,
    pub row: u8,
    /// Covalent radius (Å).
    pub r_cov: f64,
    /// Bond radius `rB0` (Å).
    pub r_b0: f64,
    /// van der Waals radius (Å).
    pub r_vdw: f64,
    /// Average atomic mass.
    pub mass: f64,
    /// Number of outer-shell electrons.
    pub n_outer: u8,
    /// Allowed valences; `-1` means "anything".
    pub valences: Vec<i32>,
}

impl ElementData {
    /// First entry of the valence list (RDKit `getDefaultValence`).
    pub fn default_valence(&self) -> i32 {
        self.valences.first().copied().unwrap_or(-1)
    }
}

static TABLE_TEXT: &str = include_str!("../data/elements.tsv");

fn table() -> &'static Vec<ElementData> {
    static TABLE: OnceLock<Vec<ElementData>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut out: Vec<ElementData> = Vec::new();
        for line in TABLE_TEXT.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 10 {
                continue;
            }
            let anum: u8 = match f[0].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if anum as usize != out.len() {
                // duplicate (legacy) names such as Uut / Uup: keep the first
                continue;
            }
            let p = |s: &str| s.parse::<f64>().unwrap_or(0.0);
            out.push(ElementData {
                atomic_num: anum,
                symbol: Box::leak(f[1].to_string().into_boxed_str()),
                row: f[2].parse().unwrap_or(0),
                r_cov: p(f[3]),
                r_b0: p(f[4]),
                r_vdw: p(f[5]),
                mass: p(f[6]),
                n_outer: f[7].parse().unwrap_or(0),
                valences: f[10..].iter().filter_map(|v| v.parse().ok()).collect(),
            });
        }
        out
    })
}

/// Returns element data for an atomic number.
///
/// # Panics
/// Panics if `anum` is larger than the table.
pub fn element(anum: u8) -> &'static ElementData {
    &table()[anum as usize]
}

/// Number of elements in the table (including the dummy atom 0).
pub fn num_elements() -> usize {
    table().len()
}

/// Looks up an atomic number by symbol (case-insensitive for the second letter).
pub fn atomic_number(symbol: &str) -> Option<u8> {
    let s = symbol.trim();
    if s.is_empty() {
        return None;
    }
    let mut norm = String::new();
    for (i, c) in s.chars().enumerate() {
        if i == 0 {
            norm.extend(c.to_uppercase());
        } else {
            norm.extend(c.to_lowercase());
        }
    }
    if norm == "D" || norm == "T" {
        return Some(1);
    }
    table().iter().find(|e| e.symbol == norm).map(|e| e.atomic_num)
}

/// RDKit `PeriodicTable::moreElectroNegative`.
pub fn more_electronegative(anum1: u8, anum2: u8) -> bool {
    let ne1 = element(anum1).n_outer;
    let ne2 = element(anum2).n_outer;
    if ne1 > ne2 {
        return true;
    }
    if ne1 == ne2 && anum1 < anum2 {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_lookup() {
        assert_eq!(element(6).symbol, "C");
        assert_eq!(element(6).default_valence(), 4);
        assert_eq!(element(7).n_outer, 5);
        assert_eq!(atomic_number("CL"), Some(17));
        assert_eq!(atomic_number("Og"), Some(118));
        assert!((element(1).r_vdw - 1.2).abs() < 1e-12);
        assert_eq!(num_elements(), 119);
    }
}
