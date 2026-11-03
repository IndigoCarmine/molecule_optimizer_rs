//! SMARTS parser.
//!
//! Recursive-descent port of the RDKit SMARTS grammar and lexer
//! (`Code/GraphMol/SmilesParse/smarts.yy`, `smarts.ll`) together with the
//! ring-closure handling of `SmilesParseOps.cpp` (`CloseMolRings`,
//! `getUnspecifiedQueryBond`).  The construction order of query atoms and
//! bonds is identical to RDKit's (atoms in input order; chain and branch
//! bonds when the second atom is read; ring-closure bonds only after the
//! whole (sub)pattern has been read, ordered by ring number), because the
//! adjacency order of the query graph determines the VF2 match order.
//!
//! Copyright (C) 2003-2026 Greg Landrum and other RDKit contributors,
//! BSD-3-Clause (see `LICENSE-RDKIT.txt`).

use super::query::{AtomExpr, AtomProp, BondExpr, Cmp, hybridization_token};
use crate::core::BondOrder;
use crate::error::{Error, Result};
use std::collections::BTreeMap;

/// A query atom.
#[derive(Debug, Clone)]
pub(crate) struct QAtom {
    pub(crate) expr: AtomExpr,
    pub(crate) map_number: Option<u32>,
}

/// A query bond.
#[derive(Debug, Clone)]
pub(crate) struct QBond {
    pub(crate) begin: usize,
    pub(crate) end: usize,
    pub(crate) expr: BondExpr,
}

/// Query graph (atoms, bonds and adjacency in bond insertion order).
#[derive(Debug, Clone, Default)]
pub(crate) struct QueryMol {
    pub(crate) atoms: Vec<QAtom>,
    pub(crate) bonds: Vec<QBond>,
    pub(crate) adj: Vec<Vec<(usize, usize)>>,
}

impl QueryMol {
    fn add_atom(&mut self, a: QAtom) -> usize {
        self.atoms.push(a);
        self.adj.push(Vec::new());
        self.atoms.len() - 1
    }
    fn has_bond(&self, a: usize, b: usize) -> bool {
        self.adj[a].iter().any(|&(n, _)| n == b)
    }
    fn add_bond(&mut self, begin: usize, end: usize, expr: BondExpr) -> usize {
        let idx = self.bonds.len();
        self.bonds.push(QBond { begin, end, expr });
        self.adj[begin].push((end, idx));
        self.adj[end].push((begin, idx));
        idx
    }
}

/// Parse result: the top-level query plus all recursive sub-patterns in
/// post order (nested patterns precede the patterns that contain them).
pub(crate) struct Parsed {
    pub(crate) mol: QueryMol,
    pub(crate) recursive: Vec<QueryMol>,
}

pub(crate) fn parse(s: &str) -> Result<Parsed> {
    // RDKit trims leading / trailing control characters and whitespace
    let s = s.trim_matches(|c: char| (c as u32) <= 32);
    if s.is_empty() {
        return Err(Error::InvalidArgument("empty SMARTS".into()));
    }
    let mut p = Parser {
        s: s.as_bytes(),
        pos: 0,
        recursive: Vec::new(),
    };
    let mol = p.parse_mol(false)?;
    if p.pos != p.s.len() {
        return Err(p.err("unexpected trailing characters"));
    }
    Ok(Parsed {
        mol,
        recursive: p.recursive,
    })
}

/// Two-letter element symbols that the RDKit lexer recognises inside
/// brackets (`ATOM_TOKEN` list in `smarts.ll`), plus the bracket-only
/// aromatic symbols and the organic-subset halogens.
const TWO_LETTER: &[&str] = &[
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Ar", "Ca", "Sc", "Ti", "Cr", "Mn", "Co", "Fe",
    "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Kr", "Rb", "Sr", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Fl", "Lv",
];

/// Element symbols not in our periodic table data are still lexed; this
/// maps them to atomic numbers directly (RDKit `PeriodicTable`).
fn element_number(sym: &str) -> Option<u8> {
    const TABLE: &[&str] = &[
        "*", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
        "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
        "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
        "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
        "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", "Uup", "Lv",
    ];
    TABLE.iter().position(|&e| e == sym).map(|p| p as u8)
}

/// Partial ring-closure bond (RDKit bond bookmark).
struct PartialBond {
    atom: usize,
    /// `None`: unspecified (`_unspecifiedOrder`), i.e. single-or-aromatic
    expr: Option<BondExpr>,
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
    recursive: Vec<QueryMol>,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> Error {
        Error::Parse {
            line: 1,
            msg: format!(
                "SMARTS {msg} at position {} in '{}'",
                self.pos,
                String::from_utf8_lossy(self.s)
            ),
        }
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }
    fn peek_at(&self, k: usize) -> Option<u8> {
        self.s.get(self.pos + k).copied()
    }
    fn starts_with(&self, t: &str) -> bool {
        self.s[self.pos..].starts_with(t.as_bytes())
    }
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, c: u8) -> Result<()> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(self.err(&format!("expected '{}'", c as char)))
        }
    }

    fn number(&mut self) -> Option<i32> {
        let start = self.pos;
        let mut v: i64 = 0;
        while let Some(c @ b'0'..=b'9') = self.peek() {
            v = v * 10 + (c - b'0') as i64;
            if v >= i32::MAX as i64 {
                self.pos = start;
                return None;
            }
            self.pos += 1;
        }
        if self.pos == start {
            None
        } else {
            Some(v as i32)
        }
    }
    fn expect_number(&mut self) -> Result<i32> {
        self.number().ok_or_else(|| self.err("expected number"))
    }

    // ------------------------------------------------------------------
    // molecule level (`mol` rule)

    fn parse_mol(&mut self, in_recursion: bool) -> Result<QueryMol> {
        let mut m = QueryMol::default();
        let first = self.parse_atomd()?;
        let mut active = m.add_atom(first);
        let mut branches: Vec<usize> = Vec::new();
        let mut rings: BTreeMap<i32, Vec<PartialBond>> = BTreeMap::new();
        loop {
            let Some(c) = self.peek() else {
                if in_recursion {
                    return Err(self.err("unterminated recursive SMARTS"));
                }
                if !branches.is_empty() {
                    return Err(self.err("unclosed branch"));
                }
                break;
            };
            match c {
                b')' => {
                    if let Some(b) = branches.pop() {
                        self.pos += 1;
                        active = b;
                    } else if in_recursion {
                        if !branches.is_empty() {
                            return Err(self.err("unclosed branch"));
                        }
                        break; // END_RECURSE consumed by the caller
                    } else {
                        return Err(self.err("extra close parenthesis"));
                    }
                }
                b'(' => {
                    self.pos += 1;
                    let bond = self.parse_bond_expr()?;
                    let a = self.parse_atomd()?;
                    let idx = m.add_atom(a);
                    let expr = bond.unwrap_or(BondExpr::SingleOrAromatic);
                    m.add_bond(active, idx, expr);
                    branches.push(active);
                    active = idx;
                }
                b'.' => {
                    self.pos += 1;
                    let a = self.parse_atomd()?;
                    active = m.add_atom(a);
                }
                _ => {
                    let bond = self.parse_bond_expr()?;
                    if let Some(rn) = self.parse_ring_number()? {
                        rings.entry(rn).or_default().push(PartialBond {
                            atom: active,
                            expr: bond,
                        });
                    } else {
                        let a = self.parse_atomd()?;
                        let idx = m.add_atom(a);
                        m.add_bond(active, idx, bond.unwrap_or(BondExpr::SingleOrAromatic));
                        active = idx;
                    }
                }
            }
        }
        // SmilesParseOps::CloseMolRings: bookmarks in ring-number order,
        // consecutive pairs connected; the first explicit bond spec wins.
        for (rn, list) in rings {
            if list.len() % 2 != 0 {
                return Err(self.err(&format!("unclosed ring {rn}")));
            }
            let mut it = list.into_iter();
            while let (Some(b1), Some(b2)) = (it.next(), it.next()) {
                if b1.atom == b2.atom {
                    return Err(self.err(&format!(
                        "ring closure {rn} bonds atom {} to itself",
                        b1.atom
                    )));
                }
                if m.has_bond(b1.atom, b2.atom) {
                    return Err(self.err(&format!("ring closure {rn} duplicates a bond")));
                }
                // the kept partial bond keeps its begin atom
                match (b1.expr, b2.expr) {
                    (Some(e), _) => m.add_bond(b1.atom, b2.atom, e),
                    (None, e) => {
                        m.add_bond(b2.atom, b1.atom, e.unwrap_or(BondExpr::SingleOrAromatic))
                    }
                };
            }
        }
        Ok(m)
    }

    fn parse_ring_number(&mut self) -> Result<Option<i32>> {
        match self.peek() {
            Some(c @ b'0'..=b'9') => {
                self.pos += 1;
                Ok(Some((c - b'0') as i32))
            }
            Some(b'%') => {
                self.pos += 1;
                if self.eat(b'(') {
                    let start = self.pos;
                    let v = self.expect_number()?;
                    if self.pos - start > 5 {
                        return Err(self.err("ring number too long"));
                    }
                    self.expect(b')')?;
                    Ok(Some(v))
                } else {
                    let d1 = self.peek().filter(|c| (b'1'..=b'9').contains(c));
                    let d2 = self.peek_at(1).filter(|c| c.is_ascii_digit());
                    match (d1, d2) {
                        (Some(a), Some(b)) => {
                            self.pos += 2;
                            Ok(Some(((a - b'0') * 10 + (b - b'0')) as i32))
                        }
                        _ => Err(self.err("bad ring number")),
                    }
                }
            }
            _ => Ok(None),
        }
    }

    // ------------------------------------------------------------------
    // atoms

    /// `atomd`: an unbracketed organic / aromatic atom, `*`, `a`, `A`, or a
    /// bracket atom.
    fn parse_atomd(&mut self) -> Result<QAtom> {
        if self.peek() == Some(b'[') {
            return self.parse_bracket_atom();
        }
        let expr = self
            .parse_simple_atom(false)?
            .ok_or_else(|| self.err("expected atom"))?;
        Ok(QAtom {
            expr,
            map_number: None,
        })
    }

    /// `simple_atom` (organic subset, aromatic subset, `*`, `a`, `A`).
    fn parse_simple_atom(&mut self, in_bracket: bool) -> Result<Option<AtomExpr>> {
        let rest = &self.s[self.pos..];
        let two = |t: &str| rest.starts_with(t.as_bytes());
        let (len, e) = if two("Cl") {
            (2, AtomExpr::atom_type(17, false))
        } else if two("Br") {
            (2, AtomExpr::atom_type(35, false))
        } else if in_bracket && two("si") {
            (2, AtomExpr::atom_type(14, true))
        } else if in_bracket && two("as") {
            (2, AtomExpr::atom_type(33, true))
        } else if in_bracket && two("se") {
            (2, AtomExpr::atom_type(34, true))
        } else if in_bracket && two("te") {
            (2, AtomExpr::atom_type(52, true))
        } else {
            match rest.first() {
                Some(b'B') => (1, AtomExpr::atom_type(5, false)),
                Some(b'C') => (1, AtomExpr::atom_type(6, false)),
                Some(b'N') => (1, AtomExpr::atom_type(7, false)),
                Some(b'O') => (1, AtomExpr::atom_type(8, false)),
                Some(b'F') => (1, AtomExpr::atom_type(9, false)),
                Some(b'P') => (1, AtomExpr::atom_type(15, false)),
                Some(b'S') => (1, AtomExpr::atom_type(16, false)),
                Some(b'I') => (1, AtomExpr::atom_type(53, false)),
                Some(b'b') => (1, AtomExpr::atom_type(5, true)),
                Some(b'c') => (1, AtomExpr::atom_type(6, true)),
                Some(b'n') => (1, AtomExpr::atom_type(7, true)),
                Some(b'o') => (1, AtomExpr::atom_type(8, true)),
                Some(b'p') => (1, AtomExpr::atom_type(15, true)),
                Some(b's') => (1, AtomExpr::atom_type(16, true)),
                Some(b'*') => (1, AtomExpr::True),
                Some(b'a') => (1, AtomExpr::prim(AtomProp::Aromatic, Cmp::Eq(1))),
                Some(b'A') => (1, AtomExpr::prim(AtomProp::Aliphatic, Cmp::Eq(1))),
                _ => return Ok(None),
            }
        };
        self.pos += len;
        Ok(Some(e))
    }

    /// Bracket element symbols that are `ATOM_TOKEN`s (atomic-number query,
    /// no aromaticity constraint).  Returns the token length.
    fn element_token(&self) -> Option<(usize, u8)> {
        let rest = &self.s[self.pos..];
        for t in ["Uut", "Uup"] {
            if rest.starts_with(t.as_bytes()) {
                return Some((3, element_number(t).unwrap()));
            }
        }
        if rest.len() >= 2 {
            let t = std::str::from_utf8(&rest[..2]).unwrap_or("");
            if TWO_LETTER.contains(&t) {
                return Some((2, element_number(t).unwrap()));
            }
        }
        match rest.first() {
            Some(&c @ (b'K' | b'V' | b'Y' | b'W' | b'U')) => Some((
                1,
                element_number(std::str::from_utf8(&[c]).unwrap()).unwrap(),
            )),
            _ => None,
        }
    }

    /// Length of the longest "letter" token at the current position inside
    /// brackets, used to mimic flex's longest-match rule between element
    /// symbols and the organic / aromatic subsets.
    fn bracket_letter_token(&self) -> Option<BracketTok> {
        let rest = &self.s[self.pos..];
        // two-letter tokens first (longest match)
        if rest.starts_with(b"Uut") || rest.starts_with(b"Uup") {
            return Some(BracketTok::Element);
        }
        if rest.starts_with(b"Cl") || rest.starts_with(b"Br") {
            return Some(BracketTok::Simple);
        }
        for t in ["si", "as", "se", "te"] {
            if rest.starts_with(t.as_bytes()) {
                return Some(BracketTok::Simple);
            }
        }
        if self.element_token().is_some_and(|(l, _)| l == 2) {
            return Some(BracketTok::Element);
        }
        match rest.first()? {
            b'B' | b'C' | b'N' | b'O' | b'F' | b'P' | b'S' | b'I' | b'b' | b'c' | b'n' | b'o'
            | b'p' | b's' | b'*' | b'a' | b'A' => Some(BracketTok::Simple),
            b'K' | b'V' | b'Y' | b'W' | b'U' => Some(BracketTok::Element),
            _ => None,
        }
    }

    fn parse_bracket_atom(&mut self) -> Result<QAtom> {
        self.expect(b'[')?;
        if let Some(a) = self.try_hydrogen_atom()? {
            return Ok(a);
        }
        let expr = self.parse_atom_semi()?;
        let map_number = if self.eat(b':') {
            let n = self.expect_number()?;
            Some(n as u32)
        } else {
            None
        };
        self.expect(b']')?;
        Ok(QAtom { expr, map_number })
    }

    /// `hydrogen_atom`: `[H]`, `[2H]`, `[H+]`, `[2H-:3]` … (an actual
    /// hydrogen atom, not an H-count query).
    fn try_hydrogen_atom(&mut self) -> Result<Option<QAtom>> {
        let save = self.pos;
        let iso = self.number();
        if !self.eat(b'H') {
            self.pos = save;
            return Ok(None);
        }
        let mut charge = None;
        if matches!(self.peek(), Some(b'+' | b'-')) {
            if self.peek_at(1) == Some(b'{') {
                self.pos = save;
                return Ok(None);
            }
            charge = Some(self.parse_charge_spec()?);
        }
        let mut map_number = None;
        if self.peek() == Some(b':') {
            self.pos += 1;
            match self.number() {
                Some(n) => map_number = Some(n as u32),
                None => {
                    self.pos = save;
                    return Ok(None);
                }
            }
        }
        if !self.eat(b']') {
            self.pos = save;
            return Ok(None);
        }
        let mut expr = AtomExpr::atomic_num(1);
        if let Some(i) = iso {
            expr = AtomExpr::and(expr, AtomExpr::prim(AtomProp::Isotope, Cmp::Eq(i)));
        }
        if let Some(c) = charge {
            expr = AtomExpr::and(expr, AtomExpr::prim(AtomProp::FormalCharge, Cmp::Eq(c)));
        }
        Ok(Some(QAtom { expr, map_number }))
    }

    /// `charge_spec` (current char is `+` or `-`).
    fn parse_charge_spec(&mut self) -> Result<i32> {
        let sign = if self.eat(b'+') {
            1
        } else if self.eat(b'-') {
            -1
        } else {
            return Err(self.err("expected charge"));
        };
        let c = if sign == 1 { b'+' } else { b'-' };
        if self.eat(c) {
            return Ok(2 * sign);
        }
        Ok(sign * self.number().unwrap_or(1))
    }

    // precedence: `;` < `,` < `&` < implicit AND < `!`
    fn parse_atom_semi(&mut self) -> Result<AtomExpr> {
        let mut e = self.parse_atom_or()?;
        while self.eat(b';') {
            let r = self.parse_atom_or()?;
            e = AtomExpr::and(e, r);
        }
        Ok(e)
    }
    fn parse_atom_or(&mut self) -> Result<AtomExpr> {
        let mut e = self.parse_atom_and()?;
        while self.eat(b',') {
            let r = self.parse_atom_and()?;
            e = AtomExpr::or(e, r);
        }
        Ok(e)
    }
    fn parse_atom_and(&mut self) -> Result<AtomExpr> {
        let mut e = self.parse_atom_implicit()?;
        while self.eat(b'&') {
            let r = self.parse_atom_implicit()?;
            e = AtomExpr::and(e, r);
        }
        Ok(e)
    }
    fn parse_atom_implicit(&mut self) -> Result<AtomExpr> {
        let mut e = self.parse_point_query()?;
        while let Some(c) = self.peek() {
            if matches!(c, b';' | b',' | b'&' | b']' | b':') {
                break;
            }
            let r = self.parse_point_query()?;
            e = AtomExpr::and(e, r);
        }
        Ok(e)
    }
    fn parse_point_query(&mut self) -> Result<AtomExpr> {
        if self.eat(b'!') {
            let e = self.parse_point_query()?;
            return Ok(AtomExpr::negate(e));
        }
        if self.starts_with("$(") {
            self.pos += 2;
            let sub = self.parse_mol(true)?;
            self.expect(b')')?;
            // undocumented serial-number extension `$(...)_nnn`
            if self.peek() == Some(b'_')
                && self.peek_at(1).is_some_and(|c| (b'1'..=b'9').contains(&c))
            {
                self.pos += 1;
                self.expect_number()?;
            }
            self.recursive.push(sub);
            return Ok(AtomExpr::Recursive(self.recursive.len() - 1));
        }
        self.parse_atom_query()
    }

    /// Optional `{lo-hi}` / `{lo-}` / `{-hi}` suffix.
    fn parse_range(&mut self) -> Result<Option<(i32, i32)>> {
        if !self.eat(b'{') {
            return Ok(None);
        }
        let (lo, hi) = if self.eat(b'-') {
            (-1, self.expect_number()?)
        } else {
            let lo = self.expect_number()?;
            self.expect(b'-')?;
            (lo, self.number().unwrap_or(-1))
        };
        self.expect(b'}')?;
        Ok(Some((lo, hi)))
    }

    fn range_cmp((lo, hi): (i32, i32)) -> Cmp {
        match (lo >= 0, hi >= 0) {
            (true, true) => Cmp::Range(lo, hi),
            (true, false) => Cmp::AtLeast(lo),
            _ => Cmp::AtMost(hi),
        }
    }

    /// `atom_query`
    fn parse_atom_query(&mut self) -> Result<AtomExpr> {
        let c = self
            .peek()
            .ok_or_else(|| self.err("unexpected end in atom"))?;
        // number-prefixed forms (isotopes)
        if c.is_ascii_digit() {
            let iso = self.expect_number()?;
            let iso_q = AtomExpr::prim(AtomProp::Isotope, Cmp::Eq(iso));
            if self.peek() == Some(b'#') {
                self.pos += 1;
                let n = self.expect_number()?;
                return Ok(AtomExpr::and(AtomExpr::atomic_num(n as u8), iso_q));
            }
            if self.peek() == Some(b'H') && self.bracket_letter_token().is_none() {
                self.pos += 1;
                let h = self.number().unwrap_or(1);
                return Ok(AtomExpr::and(
                    iso_q,
                    AtomExpr::prim(AtomProp::HCount, Cmp::Eq(h)),
                ));
            }
            if let Some(e) = self.parse_element_or_simple()? {
                return Ok(AtomExpr::and(e, iso_q));
            }
            return Ok(iso_q);
        }
        if let Some(e) = self.parse_element_or_simple()? {
            return Ok(e);
        }
        self.pos += 1;
        let e = match c {
            b'#' => {
                let n = self.expect_number()?;
                if !(0..=255).contains(&n) {
                    return Err(self.err("bad atomic number"));
                }
                AtomExpr::atomic_num(n as u8)
            }
            b'H' => {
                let h = self.number().unwrap_or(1);
                AtomExpr::prim(AtomProp::HCount, Cmp::Eq(h))
            }
            b'D' | b'd' | b'X' | b'v' | b'R' => {
                let prop = match c {
                    b'D' => AtomProp::ExplicitDegree,
                    b'd' => AtomProp::NonHydrogenDegree,
                    b'X' => AtomProp::TotalDegree,
                    b'v' => AtomProp::TotalValence,
                    _ => AtomProp::NumRings,
                };
                if let Some(r) = self.parse_range()? {
                    AtomExpr::prim(prop, Self::range_cmp(r))
                } else if let Some(n) = self.number() {
                    AtomExpr::prim(prop, Cmp::Eq(n))
                } else if c == b'R' {
                    AtomExpr::prim(prop, Cmp::NonZero)
                } else {
                    AtomExpr::prim(prop, Cmp::Eq(1))
                }
            }
            b'z' | b'Z' | b'x' | b'h' | b'r' => {
                let (num_prop, alone) = match c {
                    b'z' => (AtomProp::NumHeteroatomNbrs, AtomProp::HasHeteroatomNbrs),
                    b'Z' => (
                        AtomProp::NumAliphaticHeteroatomNbrs,
                        AtomProp::HasAliphaticHeteroatomNbrs,
                    ),
                    b'x' => (AtomProp::RingBondCount, AtomProp::HasRingBond),
                    b'h' => (AtomProp::ImplicitHCount, AtomProp::HasImplicitH),
                    _ => (AtomProp::MinRingSize, AtomProp::InRing),
                };
                if let Some(r) = self.parse_range()? {
                    AtomExpr::prim(num_prop, Self::range_cmp(r))
                } else if let Some(n) = self.number() {
                    AtomExpr::prim(num_prop, Cmp::Eq(n))
                } else {
                    AtomExpr::prim(alone, Cmp::Eq(1))
                }
            }
            b'k' => {
                if let Some((lo, hi)) = self.parse_range()? {
                    let cmp = Self::range_cmp((lo, hi));
                    AtomExpr::prim(AtomProp::RingSizeInRange(lo, hi), cmp)
                } else if let Some(n) = self.number() {
                    AtomExpr::prim(AtomProp::InRingOfSize(n), Cmp::Eq(n))
                } else {
                    AtomExpr::prim(AtomProp::InRing, Cmp::Eq(1))
                }
            }
            b'+' | b'-' => {
                if self.peek() == Some(b'{') {
                    let prop = if c == b'+' {
                        AtomProp::FormalCharge
                    } else {
                        AtomProp::NegativeFormalCharge
                    };
                    let r = self.parse_range()?.unwrap();
                    AtomExpr::prim(prop, Self::range_cmp(r))
                } else {
                    self.pos -= 1;
                    let q = self.parse_charge_spec()?;
                    AtomExpr::prim(AtomProp::FormalCharge, Cmp::Eq(q))
                }
            }
            b'@' => {
                // chirality: parsed but a null query (useChirality = false)
                let mut k = 0;
                while self.peek_at(k) == Some(b' ') {
                    k += 1;
                }
                let cls = [b"TH", b"AL", b"SP", b"TB", b"OH"]
                    .iter()
                    .any(|t| self.s[self.pos + k..].starts_with(&t[..]));
                if cls {
                    self.pos += k + 2;
                    if self.number() == Some(0) {
                        return Err(self.err("chiral permutation cannot be zero"));
                    }
                } else {
                    self.eat(b'@');
                }
                AtomExpr::True
            }
            b'^' => {
                let d = self
                    .peek()
                    .filter(|c| c.is_ascii_digit())
                    .ok_or_else(|| self.err("bad hybridization"))?;
                let h =
                    hybridization_token(d - b'0').ok_or_else(|| self.err("bad hybridization"))?;
                self.pos += 1;
                AtomExpr::prim(AtomProp::Hybridization, Cmp::Eq(h))
            }
            _ => {
                self.pos -= 1;
                return Err(self.err(&format!("unexpected character '{}'", c as char)));
            }
        };
        Ok(e)
    }

    /// Element symbol (`ATOM_TOKEN`) or `simple_atom` inside brackets,
    /// resolved with flex's longest-match rule.
    fn parse_element_or_simple(&mut self) -> Result<Option<AtomExpr>> {
        match self.bracket_letter_token() {
            Some(BracketTok::Element) => {
                let (len, num) = self.element_token().unwrap();
                self.pos += len;
                Ok(Some(AtomExpr::atomic_num(num)))
            }
            Some(BracketTok::Simple) => self.parse_simple_atom(true),
            None => Ok(None),
        }
    }

    // ------------------------------------------------------------------
    // bonds

    fn at_bond_primitive(&self) -> bool {
        match self.peek() {
            Some(b'-' | b'=' | b'#' | b':' | b'~' | b'@' | b'/' | b'\\' | b'$' | b'!') => true,
            Some(b'<') => self.peek_at(1) == Some(b'-'),
            _ => false,
        }
    }

    /// Optional bond expression before an atom or ring-closure digit.
    fn parse_bond_expr(&mut self) -> Result<Option<BondExpr>> {
        if !self.at_bond_primitive() {
            return Ok(None);
        }
        Ok(Some(self.parse_bond_semi()?))
    }
    fn parse_bond_semi(&mut self) -> Result<BondExpr> {
        let mut e = self.parse_bond_or()?;
        while self.eat(b';') {
            let r = self.parse_bond_or()?;
            e = BondExpr::And(Box::new(e), Box::new(r));
        }
        Ok(e)
    }
    fn parse_bond_or(&mut self) -> Result<BondExpr> {
        let mut e = self.parse_bond_and()?;
        while self.eat(b',') {
            let r = self.parse_bond_and()?;
            e = BondExpr::Or(Box::new(e), Box::new(r));
        }
        Ok(e)
    }
    fn parse_bond_and(&mut self) -> Result<BondExpr> {
        let mut e = self.parse_bond_query()?;
        while self.eat(b'&') {
            let r = self.parse_bond_query()?;
            e = BondExpr::And(Box::new(e), Box::new(r));
        }
        Ok(e)
    }
    /// `bond_query: bondd+` (implicit AND)
    fn parse_bond_query(&mut self) -> Result<BondExpr> {
        let mut e = self.parse_bondd()?;
        while self.at_bond_primitive() {
            let r = self.parse_bondd()?;
            e = BondExpr::And(Box::new(e), Box::new(r));
        }
        Ok(e)
    }
    fn parse_bondd(&mut self) -> Result<BondExpr> {
        let c = self.peek().ok_or_else(|| self.err("expected bond"))?;
        self.pos += 1;
        Ok(match c {
            b'!' => {
                let e = self.parse_bondd()?;
                match e {
                    BondExpr::Not(inner) => *inner,
                    other => BondExpr::Not(Box::new(other)),
                }
            }
            b'-' => {
                if self.eat(b'>') {
                    BondExpr::Order(Some(BondOrder::Dative))
                } else {
                    BondExpr::Order(Some(BondOrder::Single))
                }
            }
            b'<' => {
                self.expect(b'-')?;
                BondExpr::Order(Some(BondOrder::Dative))
            }
            b'=' => BondExpr::Order(Some(BondOrder::Double)),
            b'#' => BondExpr::Order(Some(BondOrder::Triple)),
            b':' => BondExpr::Order(Some(BondOrder::Aromatic)),
            b'$' => BondExpr::Order(None),
            b'~' => BondExpr::True,
            b'@' => BondExpr::InRing,
            b'/' => BondExpr::SingleOrAromatic,
            b'\\' => {
                self.eat(b'\\');
                BondExpr::SingleOrAromatic
            }
            _ => {
                self.pos -= 1;
                return Err(self.err(&format!("unexpected bond character '{}'", c as char)));
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BracketTok {
    Element,
    Simple,
}
