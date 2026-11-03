//! Molecular graph with 3D coordinates.

use super::element::element;
use crate::perception::rings::RingInfo;
use std::collections::BTreeMap;

/// Bond order / type (mirrors the subset of RDKit `Bond::BondType` we need).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BondOrder {
    Zero,
    Single,
    Double,
    Triple,
    Aromatic,
    /// Dative bond `begin -> end`; contributes valence only to `end`.
    Dative,
}

impl BondOrder {
    /// Bond order as a number (aromatic = 1.5).
    pub fn as_f64(self) -> f64 {
        match self {
            BondOrder::Zero => 0.0,
            BondOrder::Single | BondOrder::Dative => 1.0,
            BondOrder::Double => 2.0,
            BondOrder::Triple => 3.0,
            BondOrder::Aromatic => 1.5,
        }
    }
}

/// Hybridization (RDKit `Atom::HybridizationType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Hybridization {
    #[default]
    Unspecified,
    S,
    SP,
    SP2,
    SP3,
    SP2D,
    SP3D,
    SP3D2,
    Other,
}

/// Tetrahedral chirality tag (RDKit convention: looking from the first
/// neighbor, the remaining neighbors in bond order are CW / CCW).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChiralTag {
    #[default]
    Unspecified,
    Cw,
    Ccw,
}

/// Double-bond stereo, defined with respect to the reference atoms in
/// [`Bond::stereo_atoms`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BondStereo {
    #[default]
    None,
    Cis,
    Trans,
}

/// PDB residue information.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResidueInfo {
    pub atom_name: String,
    pub residue_name: String,
    pub residue_number: i32,
    pub chain_id: char,
    pub insertion_code: char,
    pub serial: i32,
    pub is_hetatm: bool,
    pub occupancy: f64,
    pub temp_factor: f64,
    pub alt_loc: char,
}

/// An atom.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Atom {
    pub atomic_num: u8,
    pub formal_charge: i8,
    pub isotope: u16,
    /// Hydrogens attached that are *not* represented as atoms but were
    /// explicitly given (e.g. `[nH]`).
    pub num_explicit_hs: u8,
    /// If `true`, never add implicit hydrogens to this atom.
    pub no_implicit: bool,
    pub num_radical_electrons: u8,
    /// Computed by [`crate::perception::sanitize`].
    pub implicit_hs: u8,
    pub is_aromatic: bool,
    pub hybridization: Hybridization,
    pub chiral_tag: ChiralTag,
    pub residue: Option<ResidueInfo>,
}

impl Atom {
    pub fn new(atomic_num: u8) -> Self {
        Atom { atomic_num, ..Default::default() }
    }
    pub fn symbol(&self) -> &'static str {
        element(self.atomic_num).symbol
    }
    pub fn mass(&self) -> f64 {
        element(self.atomic_num).mass
    }
    /// Hydrogens not represented as atoms (explicit + implicit counts).
    pub fn total_hidden_hs(&self) -> u32 {
        self.num_explicit_hs as u32 + self.implicit_hs as u32
    }
}

/// A bond between atoms `begin` and `end`.
#[derive(Debug, Clone, PartialEq)]
pub struct Bond {
    pub begin: usize,
    pub end: usize,
    pub order: BondOrder,
    pub is_aromatic: bool,
    pub is_conjugated: bool,
    pub stereo: BondStereo,
    /// Reference atoms for [`BondStereo`] (neighbor of `begin`, neighbor of `end`).
    pub stereo_atoms: Option<(usize, usize)>,
}

impl Bond {
    pub fn new(begin: usize, end: usize, order: BondOrder) -> Self {
        Bond {
            begin,
            end,
            order,
            is_aromatic: order == BondOrder::Aromatic,
            is_conjugated: false,
            stereo: BondStereo::None,
            stereo_atoms: None,
        }
    }
    pub fn other(&self, idx: usize) -> usize {
        if self.begin == idx { self.end } else { self.begin }
    }
    pub fn contains(&self, idx: usize) -> bool {
        self.begin == idx || self.end == idx
    }
    /// RDKit `Bond::getValenceContrib(atom)`.
    pub fn valence_contrib(&self, atom: usize) -> f64 {
        match self.order {
            BondOrder::Dative => {
                if atom == self.end { 1.0 } else { 0.0 }
            }
            o => o.as_f64(),
        }
    }
}

/// Molecule: atoms, bonds, adjacency and coordinates (Å).
#[derive(Debug, Clone, Default)]
pub struct Molecule {
    pub name: String,
    atoms: Vec<Atom>,
    bonds: Vec<Bond>,
    /// adjacency: for each atom, list of (neighbor, bond index)
    adj: Vec<Vec<(usize, usize)>>,
    /// one position per atom
    pub coords: Vec<[f64; 3]>,
    /// Ring perception result (filled by [`crate::perception::sanitize`]).
    pub ring_info: Option<RingInfo>,
    /// Free-form properties (e.g. SDF data fields).
    pub props: BTreeMap<String, String>,
}

impl Molecule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_atoms(&self) -> usize {
        self.atoms.len()
    }
    pub fn num_bonds(&self) -> usize {
        self.bonds.len()
    }
    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }
    pub fn atoms_mut(&mut self) -> &mut [Atom] {
        &mut self.atoms
    }
    pub fn bonds(&self) -> &[Bond] {
        &self.bonds
    }
    pub fn bonds_mut(&mut self) -> &mut [Bond] {
        &mut self.bonds
    }
    pub fn atom(&self, i: usize) -> &Atom {
        &self.atoms[i]
    }
    pub fn atom_mut(&mut self, i: usize) -> &mut Atom {
        &mut self.atoms[i]
    }
    pub fn bond(&self, i: usize) -> &Bond {
        &self.bonds[i]
    }
    pub fn bond_mut(&mut self, i: usize) -> &mut Bond {
        &mut self.bonds[i]
    }

    /// Adds an atom at `pos` and returns its index.
    pub fn add_atom(&mut self, atom: Atom, pos: [f64; 3]) -> usize {
        self.atoms.push(atom);
        self.adj.push(Vec::new());
        self.coords.push(pos);
        self.ring_info = None;
        self.atoms.len() - 1
    }

    /// Adds a bond and returns its index. Duplicate bonds are rejected
    /// (the existing index is returned).
    pub fn add_bond(&mut self, begin: usize, end: usize, order: BondOrder) -> usize {
        assert!(begin < self.atoms.len() && end < self.atoms.len() && begin != end);
        if let Some(b) = self.bond_between(begin, end) {
            return b;
        }
        let idx = self.bonds.len();
        self.bonds.push(Bond::new(begin, end, order));
        self.adj[begin].push((end, idx));
        self.adj[end].push((begin, idx));
        self.ring_info = None;
        idx
    }

    /// Removes all bonds (used by bond perception).
    pub fn clear_bonds(&mut self) {
        self.bonds.clear();
        for a in &mut self.adj {
            a.clear();
        }
        self.ring_info = None;
    }

    /// Neighbor atom indices of `i`.
    pub fn neighbors(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        self.adj[i].iter().map(|&(n, _)| n)
    }
    /// (neighbor, bond index) pairs of atom `i`.
    pub fn atom_bonds(&self, i: usize) -> &[(usize, usize)] {
        &self.adj[i]
    }
    /// Number of explicit neighbors (RDKit `getDegree`).
    pub fn degree(&self, i: usize) -> usize {
        self.adj[i].len()
    }
    /// Degree including hidden hydrogens (RDKit `getTotalDegree`).
    pub fn total_degree(&self, i: usize) -> usize {
        self.adj[i].len() + self.atoms[i].total_hidden_hs() as usize
    }
    /// Total number of hydrogens: hydrogen neighbors + hidden hydrogens
    /// (RDKit `getTotalNumHs(true)`).
    pub fn total_num_hs(&self, i: usize) -> usize {
        self.neighbors(i).filter(|&n| self.atoms[n].atomic_num == 1).count()
            + self.atoms[i].total_hidden_hs() as usize
    }

    pub fn bond_between(&self, a: usize, b: usize) -> Option<usize> {
        self.adj[a].iter().find(|&&(n, _)| n == b).map(|&(_, bi)| bi)
    }

    /// Explicit valence (RDKit `getValence(EXPLICIT)`): sum of bond valence
    /// contributions plus explicit hidden Hs, aromatic sums rounded.
    pub fn explicit_valence(&self, i: usize) -> i32 {
        let mut acc = 0.0;
        for &(_, bi) in &self.adj[i] {
            acc += self.bonds[bi].valence_contrib(i);
        }
        acc += self.atoms[i].num_explicit_hs as f64;
        (acc + 0.1).floor() as i32
    }
    /// Total valence = explicit valence + implicit Hs.
    pub fn total_valence(&self, i: usize) -> i32 {
        self.explicit_valence(i) + self.atoms[i].implicit_hs as i32
    }

    pub fn pos(&self, i: usize) -> [f64; 3] {
        self.coords[i]
    }
    pub fn set_pos(&mut self, i: usize, p: [f64; 3]) {
        self.coords[i] = p;
    }
    /// Coordinates as a flat `[x0,y0,z0,x1,...]` vector.
    pub fn flat_coords(&self) -> Vec<f64> {
        self.coords.iter().flat_map(|p| p.iter().copied()).collect()
    }
    /// Sets coordinates from a flat vector (dimension 3, or larger: extra
    /// dimensions are dropped).
    pub fn set_flat_coords(&mut self, flat: &[f64]) {
        let n = self.atoms.len();
        let dim = if n == 0 { 3 } else { flat.len() / n };
        assert!(dim >= 3, "flat coordinate vector too short");
        for i in 0..n {
            self.coords[i] = [flat[dim * i], flat[dim * i + 1], flat[dim * i + 2]];
        }
    }

    /// Heavy (non-hydrogen) atom indices.
    pub fn heavy_atoms(&self) -> Vec<usize> {
        (0..self.atoms.len()).filter(|&i| self.atoms[i].atomic_num > 1).collect()
    }

    /// `true` if any atom carries hidden hydrogens.
    pub fn has_hidden_hydrogens(&self) -> bool {
        self.atoms.iter().any(|a| a.total_hidden_hs() > 0)
    }

    /// Topological distance matrix (number of bonds; `usize::MAX` if disconnected).
    pub fn topological_distances(&self) -> Vec<Vec<usize>> {
        let n = self.atoms.len();
        let mut d = vec![vec![usize::MAX; n]; n];
        let mut queue = std::collections::VecDeque::new();
        for s in 0..n {
            d[s][s] = 0;
            queue.clear();
            queue.push_back(s);
            while let Some(u) = queue.pop_front() {
                let du = d[s][u];
                for &(v, _) in &self.adj[u] {
                    if d[s][v] == usize::MAX {
                        d[s][v] = du + 1;
                        queue.push_back(v);
                    }
                }
            }
        }
        d
    }

    /// Shortest bond path between two atoms (inclusive), if connected.
    pub fn shortest_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        let n = self.atoms.len();
        let mut prev = vec![usize::MAX; n];
        let mut seen = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        seen[from] = true;
        q.push_back(from);
        while let Some(u) = q.pop_front() {
            if u == to {
                break;
            }
            for &(v, _) in &self.adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    prev[v] = u;
                    q.push_back(v);
                }
            }
        }
        if !seen[to] {
            return None;
        }
        let mut path = vec![to];
        let mut c = to;
        while c != from {
            c = prev[c];
            path.push(c);
        }
        path.reverse();
        Some(path)
    }

    /// Atoms reachable from `start` without crossing the bond `start`-`blocked`.
    /// Returns `None` if `blocked` is reachable (i.e. the bond is in a ring).
    pub fn fragment_side(&self, start: usize, blocked: usize) -> Option<Vec<usize>> {
        let n = self.atoms.len();
        let mut seen = vec![false; n];
        seen[start] = true;
        let mut stack = vec![start];
        let mut out = vec![start];
        while let Some(u) = stack.pop() {
            for &(v, _) in &self.adj[u] {
                if u == start && v == blocked {
                    continue;
                }
                if v == blocked {
                    return None;
                }
                if !seen[v] {
                    seen[v] = true;
                    out.push(v);
                    stack.push(v);
                }
            }
        }
        Some(out)
    }

    /// Connected components (lists of atom indices).
    pub fn fragments(&self) -> Vec<Vec<usize>> {
        let n = self.atoms.len();
        let mut comp = vec![usize::MAX; n];
        let mut out = Vec::new();
        for s in 0..n {
            if comp[s] != usize::MAX {
                continue;
            }
            let id = out.len();
            let mut members = vec![s];
            comp[s] = id;
            let mut k = 0;
            while k < members.len() {
                let u = members[k];
                k += 1;
                for &(v, _) in &self.adj[u] {
                    if comp[v] == usize::MAX {
                        comp[v] = id;
                        members.push(v);
                    }
                }
            }
            out.push(members);
        }
        out
    }

    /// Ring info, computing it on demand if needed.
    pub fn rings(&mut self) -> &RingInfo {
        if self.ring_info.is_none() {
            self.ring_info = Some(RingInfo::compute(self));
        }
        self.ring_info.as_ref().unwrap()
    }

    /// Ring info if already computed.
    pub fn ring_info(&self) -> Option<&RingInfo> {
        self.ring_info.as_ref()
    }

    /// Molecular formula in Hill order.
    pub fn formula(&self) -> String {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut nh = 0usize;
        for a in &self.atoms {
            if a.atomic_num == 1 {
                nh += 1;
            } else {
                *counts.entry(a.symbol()).or_default() += 1;
            }
            nh += a.total_hidden_hs() as usize;
        }
        let mut s = String::new();
        let mut push = |sym: &str, c: usize| {
            if c > 0 {
                s.push_str(sym);
                if c > 1 {
                    s.push_str(&c.to_string());
                }
            }
        };
        if let Some(c) = counts.remove("C") {
            push("C", c);
            push("H", nh);
            nh = 0;
        }
        let mut rest: Vec<_> = counts.into_iter().collect();
        if nh > 0 {
            rest.push(("H", nh));
            rest.sort();
        }
        for (sym, c) in rest {
            push(sym, c);
        }
        s
    }
}
