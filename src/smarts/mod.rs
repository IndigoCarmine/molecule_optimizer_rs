//! SMARTS subset parser and substructure matcher (used by ETKDG torsion
//! preferences).
//!
//! Port of RDKit's SMARTS parser (`Code/GraphMol/SmilesParse/smarts.yy`,
//! `smarts.ll`, `SmilesParseOps.cpp`), query primitives
//! (`Code/GraphMol/QueryOps.{h,cpp}`) and VF2 substructure matcher
//! (`Code/GraphMol/Substruct/vf2.hpp`, `SubstructMatch.cpp`,
//! `SubstructUtils.cpp`).  Matches are enumerated in exactly RDKit's order
//! (`GetSubstructMatches(..., useChirality=False)`).
//!
//! Copyright (C) 2001-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause (see `LICENSE-RDKIT.txt`).
//!
//! Semantics worth knowing (all as in RDKit):
//! * `C`/`c` are aliphatic/aromatic element queries, bracketed non-organic
//!   symbols (`[Si]`) only test the atomic number, `#n` the atomic number;
//! * `H<n>` is the total H count including explicit H neighbours, `h<n>`
//!   counts hidden (implicit + explicit-property) Hs only;
//! * `R` = in any ring, `R<n>` = in exactly n SSSR rings, `r` = in a ring,
//!   `r<n>` = smallest ring size n, `r{a-b}` ranges over the smallest ring
//!   size (0 for acyclic atoms, so `r{-8}` also matches chain atoms),
//!   `x<n>` = number of ring bonds, `x` = has a ring bond;
//! * an omitted bond means single-or-aromatic; `@` on a bond = ring bond;
//! * chirality (`@`, `@@`) is parsed and ignored for matching;
//! * a recursive query `$(...)` matches an atom if the sub-pattern matches
//!   with its first atom on that atom.

mod parser;
mod query;
mod vf2;

use crate::core::Molecule;
use crate::error::Result;
use query::Target;
use std::collections::HashSet;

/// A parsed SMARTS pattern.
#[derive(Debug, Clone)]
pub struct Smarts {
    text: String,
    mol: parser::QueryMol,
    /// recursive sub-patterns in post order
    recursive: Vec<parser::QueryMol>,
}

impl Smarts {
    /// Parses a SMARTS string.
    pub fn parse(s: &str) -> Result<Smarts> {
        let p = parser::parse(s)?;
        Ok(Smarts {
            text: s.to_string(),
            mol: p.mol,
            recursive: p.recursive,
        })
    }

    /// The SMARTS text this pattern was parsed from.
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Number of query atoms.
    pub fn num_atoms(&self) -> usize {
        self.mol.atoms.len()
    }

    /// Number of query bonds.
    pub fn num_bonds(&self) -> usize {
        self.mol.bonds.len()
    }

    /// Query bonds as `(begin, end)` atom indices in RDKit bond order.
    pub fn bond_atoms(&self) -> Vec<(usize, usize)> {
        self.mol.bonds.iter().map(|b| (b.begin, b.end)).collect()
    }

    /// Query atom index carrying atom-map number `n` (e.g. `[C:2]`).
    pub fn atom_with_map_number(&self, n: u32) -> Option<usize> {
        self.mol.atoms.iter().position(|a| a.map_number == Some(n))
    }

    /// Atom-map number of every query atom.
    pub fn atom_map_numbers(&self) -> Vec<Option<u32>> {
        self.mol.atoms.iter().map(|a| a.map_number).collect()
    }

    /// Matches, each indexed by query atom index -> molecule atom index.
    ///
    /// * `unique == false`: every match, in RDKit's VF2 order
    ///   (`SubstructMatch` with `uniquify=false`);
    /// * `unique == true`: RDKit uniquify semantics (a match is dropped if
    ///   its atom *set* was already seen; the first one is kept).
    ///
    /// The molecule should be sanitized; if it carries no ring information
    /// it is computed locally.
    pub fn find_matches(&self, mol: &Molecule, unique: bool) -> Vec<Vec<usize>> {
        let t = Target::new(mol);
        let rec = vf2::recursive_sets(&self.recursive, &t);
        let all = vf2::all_matches(&self.mol, &t, &rec);
        if !unique {
            return all;
        }
        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        all.into_iter()
            .filter(|m| {
                let mut key = m.clone();
                key.sort_unstable();
                seen.insert(key)
            })
            .collect()
    }

    /// `true` if the pattern matches `mol` at least once.
    pub fn has_match(&self, mol: &Molecule) -> bool {
        let t = Target::new(mol);
        let rec = vf2::recursive_sets(&self.recursive, &t);
        vf2::any_match(&self.mol, &t, &rec)
    }
}

impl std::str::FromStr for Smarts {
    type Err = crate::error::Error;
    fn from_str(s: &str) -> Result<Smarts> {
        Smarts::parse(s)
    }
}

#[cfg(test)]
mod tests;
