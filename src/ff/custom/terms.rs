//! Custom term specifications and their [`Contrib`] implementations.

use super::kernel::{Globals, Kernel, lower_statement};
use super::parse::{self, ExprError};
use super::program::Program;
use super::{intern_label, ir::Graph};
use crate::core::Molecule;
use crate::core::geometry::{cross, dot, norm, scale, sub};
use crate::error::{Error, Result};
use crate::ff::Contrib;
use crate::ff::constraints::torsion_grad;
use crate::ff::utils::{add_grad, compute_dihedral, p3};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Specifications
// ---------------------------------------------------------------------------

/// A bond term `f(r)` (Å) on explicit atom pairs.
///
/// ```
/// use molecule_optimizer_rs::ff::custom::CustomBond;
/// let t = CustomBond::new("morse", "D*(1 - exp(-a*(r - r0)))^2", &["D", "a", "r0"])
///     .term(0, 1, &[100.0, 2.0, 1.1]);
/// ```
#[derive(Debug, Clone)]
pub struct CustomBond {
    pub name: String,
    pub expr: String,
    pub params: Vec<String>,
    pub terms: Vec<([usize; 2], Vec<f64>)>,
}

/// An angle term `f(theta)` (radians) on explicit atom triples `i-j-k`
/// (`j` is the vertex).
#[derive(Debug, Clone)]
pub struct CustomAngle {
    pub name: String,
    pub expr: String,
    pub params: Vec<String>,
    pub terms: Vec<([usize; 3], Vec<f64>)>,
}

/// A torsion term `f(phi)` (radians, RDKit sign convention, in `(-pi, pi]`)
/// on explicit atom quadruples `i-j-k-l`.
#[derive(Debug, Clone)]
pub struct CustomTorsion {
    pub name: String,
    pub expr: String,
    pub params: Vec<String>,
    pub terms: Vec<([usize; 4], Vec<f64>)>,
}

/// An external field / restraint `f(x, y, z)` (Å) applied to single atoms.
#[derive(Debug, Clone)]
pub struct CustomExternal {
    pub name: String,
    pub expr: String,
    pub params: Vec<String>,
    pub atoms: Vec<(usize, Vec<f64>)>,
}

macro_rules! bonded_spec {
    ($t:ident, $n:literal, $($a:ident),+) => {
        impl $t {
            pub fn new(name: &str, expr: &str, params: &[&str]) -> Self {
                $t { name: name.to_string(), expr: expr.to_string(), params: params.iter().map(|s| s.to_string()).collect(), terms: Vec::new() }
            }
            /// Adds one term with its parameter values (in the order of `params`).
            pub fn term(mut self, $($a: usize),+, values: &[f64]) -> Self {
                self.terms.push(([$($a),+], values.to_vec()));
                self
            }
            pub fn add_term(&mut self, atoms: [usize; $n], values: &[f64]) {
                self.terms.push((atoms, values.to_vec()));
            }
        }
    };
}
bonded_spec!(CustomBond, 2, i, j);
bonded_spec!(CustomAngle, 3, i, j, k);
bonded_spec!(CustomTorsion, 4, i, j, k, l);

impl CustomExternal {
    pub fn new(name: &str, expr: &str, params: &[&str]) -> Self {
        CustomExternal {
            name: name.to_string(),
            expr: expr.to_string(),
            params: params.iter().map(|s| s.to_string()).collect(),
            atoms: Vec::new(),
        }
    }
    /// Applies the field to atom `i` with the given parameter values.
    pub fn atom(mut self, i: usize, values: &[f64]) -> Self {
        self.atoms.push((i, values.to_vec()));
        self
    }
    /// Applies the field to every atom in `atoms` with the same values.
    pub fn atoms(mut self, atoms: impl IntoIterator<Item = usize>, values: &[f64]) -> Self {
        self.atoms.extend(atoms.into_iter().map(|i| (i, values.to_vec())));
        self
    }
}

/// Per-atom parameter function: `f(mol, atom) -> values`.
pub type AtomParamFn = Arc<dyn Fn(&Molecule, usize) -> Vec<f64> + Send + Sync>;

/// Values of the per-atom parameters of a [`CustomNonbonded`] term.
#[derive(Clone)]
pub enum AtomParams {
    /// The same values for every atom.
    Uniform(Vec<f64>),
    /// One row per atom (must cover all atoms of the molecule).
    Table(Vec<Vec<f64>>),
    /// Computed from the molecule (e.g. by element).
    Function(AtomParamFn),
}

impl std::fmt::Debug for AtomParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomParams::Uniform(v) => f.debug_tuple("Uniform").field(v).finish(),
            AtomParams::Table(t) => f.debug_tuple("Table").field(&t.len()).finish(),
            AtomParams::Function(_) => f.write_str("Function(..)"),
        }
    }
}

/// Cutoff of a nonbonded term: the energy is zero for `r >= cutoff`. With
/// `switch_from = Some(rs)` the energy is multiplied by the quintic switch
/// `S(t) = 1 - 10t^3 + 15t^4 - 6t^5`, `t = (r - rs)/(cutoff - rs)`, so that
/// energy and force go smoothly to zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cutoff {
    pub cutoff: f64,
    pub switch_from: Option<f64>,
}

/// Which pairs a nonbonded term acts on.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum InteractionGroups {
    /// All pairs.
    #[default]
    All,
    /// Only pairs of atoms in different fragments (connected components).
    InterFragment,
    /// Only pairs of atoms in the same fragment.
    IntraFragment,
    /// Only pairs `(a, b)` with `a` in the first set and `b` in the second;
    /// `a` is atom "1" and `b` atom "2" in the combining rules.
    Between(Vec<usize>, Vec<usize>),
}

/// A pairwise nonbonded term `f(r)`.
///
/// Per-atom parameters `p` appear in the expressions as `p1` (first atom)
/// and `p2` (second atom). Combining rules are definitions of per-pair
/// parameters in terms of those (`"sig = 0.5*(sig1+sig2); eps = sqrt(eps1*eps2)"`);
/// they are evaluated once per pair at build time and may be overridden for
/// specific pairs with [`CustomNonbonded::pair_override`].
#[derive(Debug, Clone)]
pub struct CustomNonbonded {
    pub name: String,
    pub expr: String,
    pub per_atom_params: Vec<String>,
    pub combining: String,
    pub atom_params: AtomParams,
    /// Explicit values of the combined parameters for specific pairs.
    pub pair_overrides: Vec<((usize, usize), Vec<f64>)>,
    pub cutoff: Option<Cutoff>,
    /// Pairs separated by at most this many bonds are excluded
    /// (0 = none, 1 = bonded pairs, 3 = 1-2, 1-3 and 1-4 pairs).
    pub exclude_within_bonds: usize,
    /// Additional excluded pairs (unordered).
    pub exclusions: Vec<(usize, usize)>,
    pub groups: InteractionGroups,
}

impl CustomNonbonded {
    pub fn new(name: &str, expr: &str) -> Self {
        CustomNonbonded {
            name: name.to_string(),
            expr: expr.to_string(),
            per_atom_params: Vec::new(),
            combining: String::new(),
            atom_params: AtomParams::Uniform(Vec::new()),
            pair_overrides: Vec::new(),
            cutoff: None,
            exclude_within_bonds: 0,
            exclusions: Vec::new(),
            groups: InteractionGroups::All,
        }
    }
    /// Declares per-atom parameters and their values.
    pub fn per_atom(mut self, names: &[&str], values: AtomParams) -> Self {
        self.per_atom_params = names.iter().map(|s| s.to_string()).collect();
        self.atom_params = values;
        self
    }
    /// Combining rules (definitions of per-pair parameters).
    pub fn combining(mut self, rules: &str) -> Self {
        self.combining = rules.to_string();
        self
    }
    pub fn pair_override(mut self, i: usize, j: usize, values: &[f64]) -> Self {
        self.pair_overrides.push(((i, j), values.to_vec()));
        self
    }
    pub fn cutoff(mut self, cutoff: f64, switch_from: Option<f64>) -> Self {
        self.cutoff = Some(Cutoff { cutoff, switch_from });
        self
    }
    pub fn exclude_within_bonds(mut self, n: usize) -> Self {
        self.exclude_within_bonds = n;
        self
    }
    pub fn exclude(mut self, i: usize, j: usize) -> Self {
        self.exclusions.push((i, j));
        self
    }
    pub fn groups(mut self, g: InteractionGroups) -> Self {
        self.groups = g;
        self
    }
}

/// Any custom term.
#[derive(Debug, Clone)]
pub enum CustomTerm {
    Nonbonded(CustomNonbonded),
    Bond(CustomBond),
    Angle(CustomAngle),
    Torsion(CustomTorsion),
    External(CustomExternal),
}

macro_rules! from_term {
    ($t:ident, $v:ident) => {
        impl From<$t> for CustomTerm {
            fn from(t: $t) -> Self {
                CustomTerm::$v(t)
            }
        }
    };
}
from_term!(CustomNonbonded, Nonbonded);
from_term!(CustomBond, Bond);
from_term!(CustomAngle, Angle);
from_term!(CustomTorsion, Torsion);
from_term!(CustomExternal, External);

impl CustomTerm {
    pub fn name(&self) -> &str {
        match self {
            CustomTerm::Nonbonded(t) => &t.name,
            CustomTerm::Bond(t) => &t.name,
            CustomTerm::Angle(t) => &t.name,
            CustomTerm::Torsion(t) => &t.name,
            CustomTerm::External(t) => &t.name,
        }
    }
}

// ---------------------------------------------------------------------------
// Prepared (compiled) terms
// ---------------------------------------------------------------------------

/// A custom term with its expressions compiled.
#[derive(Debug, Clone)]
pub struct PreparedTerm {
    pub spec: CustomTerm,
    pub kernel: Arc<Kernel>,
    /// Nonbonded only: `[p1.. | p2..]` -> combined parameters.
    combine: Option<Program<f64>>,
    n_combined: usize,
    label: &'static str,
}

fn strs(v: &[String]) -> Vec<&str> {
    v.iter().map(|s| s.as_str()).collect()
}

impl PreparedTerm {
    /// Parses and compiles the expressions of `spec` against `globals`.
    pub fn new(spec: CustomTerm, globals: &Globals) -> std::result::Result<PreparedTerm, ExprError> {
        let label = intern_label(spec.name());
        let (kernel, combine, n_combined) = match &spec {
            CustomTerm::Bond(t) => (Kernel::compile(&t.expr, &["r"], &strs(&t.params), globals)?, None, 0),
            CustomTerm::Angle(t) => (Kernel::compile(&t.expr, &["theta"], &strs(&t.params), globals)?, None, 0),
            CustomTerm::Torsion(t) => (Kernel::compile(&t.expr, &["phi"], &strs(&t.params), globals)?, None, 0),
            CustomTerm::External(t) => {
                (Kernel::compile(&t.expr, &["x", "y", "z"], &strs(&t.params), globals)?, None, 0)
            }
            CustomTerm::Nonbonded(t) => {
                let mut atom_vars: Vec<String> = t.per_atom_params.iter().map(|p| format!("{p}1")).collect();
                atom_vars.extend(t.per_atom_params.iter().map(|p| format!("{p}2")));
                let rules = parse::parse_definitions(&t.combining)?;
                let combined: Vec<String> =
                    rules.statements.iter().map(|s| s.name.as_ref().unwrap().0.clone()).collect();
                let mut g = Graph::new();
                let mut roots = Vec::new();
                for k in 0..rules.statements.len() {
                    let mut lookup = |n: &str| atom_vars.iter().position(|v| v == n).map(|i| i as u32);
                    roots.push(lower_statement(&rules, k, &mut g, &mut lookup)?);
                }
                let combine = Program::compile(&g, &roots, atom_vars.len());
                let mut params = atom_vars.clone();
                params.extend(combined.iter().cloned());
                let kernel = Kernel::compile(&t.expr, &["r"], &strs(&params), globals)?;
                (kernel, Some(combine), combined.len())
            }
        };
        Ok(PreparedTerm { spec, kernel: Arc::new(kernel), combine, n_combined, label })
    }

    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Creates the contrib for `mol`.
    pub fn contrib(&self, mol: &Molecule, globals: &Globals) -> Result<Box<dyn Contrib>> {
        let n = mol.num_atoms();
        let k = &self.kernel;
        let np = k.n_params();
        let nh = k.n_hoisted();
        let name = self.spec.name();
        let check_atoms = |atoms: &[usize]| -> Result<()> {
            match atoms.iter().find(|&&a| a >= n) {
                Some(a) => Err(Error::InvalidArgument(format!(
                    "custom term '{name}': atom index {a} out of range ({n} atoms)"
                ))),
                None => Ok(()),
            }
        };
        let check_values = |v: &[f64]| -> Result<()> {
            if v.len() != np {
                return Err(Error::InvalidArgument(format!(
                    "custom term '{name}': {} parameter values given, {np} expected",
                    v.len()
                )));
            }
            Ok(())
        };
        let bonded = |kind: BondedKind, terms: Vec<(&[usize], &[f64])>| -> Result<Box<dyn Contrib>> {
            let mut atoms = Vec::with_capacity(terms.len() * kind.arity());
            let mut hoisted = vec![0.0; terms.len() * nh];
            for (t, (a, v)) in terms.iter().enumerate() {
                check_atoms(a)?;
                check_values(v)?;
                atoms.extend_from_slice(a);
                k.hoist(v, &mut hoisted[t * nh..(t + 1) * nh]);
            }
            Ok(Box::new(CustomBondedContrib {
                label: self.label,
                kind,
                kernel: k.clone(),
                globals: globals.clone(),
                atoms,
                hoisted,
            }))
        };
        match &self.spec {
            CustomTerm::Bond(t) => bonded(BondedKind::Bond, t.terms.iter().map(|(a, v)| (&a[..], &v[..])).collect()),
            CustomTerm::Angle(t) => bonded(BondedKind::Angle, t.terms.iter().map(|(a, v)| (&a[..], &v[..])).collect()),
            CustomTerm::Torsion(t) => {
                bonded(BondedKind::Torsion, t.terms.iter().map(|(a, v)| (&a[..], &v[..])).collect())
            }
            CustomTerm::External(t) => {
                bonded(BondedKind::External, t.atoms.iter().map(|(a, v)| (std::slice::from_ref(a), &v[..])).collect())
            }
            CustomTerm::Nonbonded(t) => self.nonbonded_contrib(t, mol, globals),
        }
    }

    fn nonbonded_contrib(&self, t: &CustomNonbonded, mol: &Molecule, globals: &Globals) -> Result<Box<dyn Contrib>> {
        let n = mol.num_atoms();
        let name = &t.name;
        let npa = t.per_atom_params.len();
        let bad = |m: String| Error::InvalidArgument(format!("custom nonbonded '{name}': {m}"));
        // Per-atom values.
        let mut atom_vals = vec![0.0; n * npa];
        for a in 0..n {
            let row: Vec<f64> = match &t.atom_params {
                AtomParams::Uniform(v) => v.clone(),
                AtomParams::Table(tab) => {
                    tab.get(a).cloned().ok_or_else(|| bad(format!("no parameters for atom {a}")))?
                }
                AtomParams::Function(f) => f(mol, a),
            };
            if row.len() != npa {
                return Err(bad(format!("atom {a} has {} parameter values, {npa} expected", row.len())));
            }
            atom_vals[a * npa..(a + 1) * npa].copy_from_slice(&row);
        }
        if let Some(c) = t.cutoff {
            if c.cutoff.is_nan() || c.cutoff <= 0.0 {
                return Err(bad("cutoff must be positive".into()));
            }
            if let Some(rs) = c.switch_from
                && !(rs >= 0.0 && rs < c.cutoff)
            {
                return Err(bad("switching distance must be in [0, cutoff)".into()));
            }
        }
        // Pair list.
        let mut excluded: HashSet<(usize, usize)> = HashSet::new();
        let key = |i: usize, j: usize| if i < j { (i, j) } else { (j, i) };
        for &(i, j) in &t.exclusions {
            if i >= n || j >= n {
                return Err(bad(format!("excluded pair ({i}, {j}) out of range")));
            }
            excluded.insert(key(i, j));
        }
        if t.exclude_within_bonds > 0 {
            // Depth-limited BFS from every atom.
            let mut depth = vec![usize::MAX; n];
            let mut visited = Vec::new();
            for s in 0..n {
                depth[s] = 0;
                visited.clear();
                visited.push(s);
                let mut k = 0;
                while k < visited.len() {
                    let u = visited[k];
                    k += 1;
                    if depth[u] == t.exclude_within_bonds {
                        continue;
                    }
                    for v in mol.neighbors(u) {
                        if depth[v] == usize::MAX {
                            depth[v] = depth[u] + 1;
                            visited.push(v);
                        }
                    }
                }
                for &v in &visited {
                    if v > s {
                        excluded.insert((s, v));
                    }
                    depth[v] = usize::MAX;
                }
            }
        }
        let frag = || {
            let mut f = vec![0usize; n];
            for (id, members) in mol.fragments().iter().enumerate() {
                for &a in members {
                    f[a] = id;
                }
            }
            f
        };
        let mut pairs: Vec<[u32; 2]> = Vec::new();
        let push = |i: usize, j: usize, pairs: &mut Vec<[u32; 2]>| {
            if i != j && !excluded.contains(&key(i, j)) {
                pairs.push([i as u32, j as u32]);
            }
        };
        match &t.groups {
            InteractionGroups::All => (0..n).for_each(|i| (i + 1..n).for_each(|j| push(i, j, &mut pairs))),
            InteractionGroups::InterFragment | InteractionGroups::IntraFragment => {
                let f = frag();
                let inter = t.groups == InteractionGroups::InterFragment;
                for i in 0..n {
                    for j in i + 1..n {
                        if (f[i] != f[j]) == inter {
                            push(i, j, &mut pairs);
                        }
                    }
                }
            }
            InteractionGroups::Between(a, b) => {
                let mut seen = HashSet::new();
                for &i in a {
                    for &j in b {
                        if i >= n || j >= n {
                            return Err(bad(format!("group atom index out of range ({n} atoms)")));
                        }
                        if i != j && seen.insert(key(i, j)) {
                            push(i, j, &mut pairs);
                        }
                    }
                }
            }
        }
        // Per-pair parameters: [p1.. | p2.. | combined..] -> hoisted.
        let k = &self.kernel;
        let nh = k.n_hoisted();
        let mut overrides = std::collections::HashMap::new();
        for ((i, j), v) in &t.pair_overrides {
            if self.n_combined == 0 {
                return Err(bad("pair overrides require combining rules (they set the combined parameters)".into()));
            }
            if v.len() != self.n_combined {
                return Err(bad(format!(
                    "pair override ({i}, {j}) has {} values, {} expected",
                    v.len(),
                    self.n_combined
                )));
            }
            overrides.insert(key(*i, *j), v.clone());
        }
        let combine = self.combine.as_ref().unwrap();
        let mut raw = vec![0.0; 2 * npa + self.n_combined];
        let mut hoisted = vec![0.0; pairs.len() * nh];
        for (p, &[i, j]) in pairs.iter().enumerate() {
            let (i, j) = (i as usize, j as usize);
            raw[..npa].copy_from_slice(&atom_vals[i * npa..(i + 1) * npa]);
            raw[npa..2 * npa].copy_from_slice(&atom_vals[j * npa..(j + 1) * npa]);
            let (head, tail) = raw.split_at_mut(2 * npa);
            match overrides.get(&key(i, j)) {
                Some(v) => tail.copy_from_slice(v),
                None => combine.eval(head, tail),
            }
            k.hoist(&raw, &mut hoisted[p * nh..(p + 1) * nh]);
        }
        Ok(Box::new(CustomNonbondedContrib {
            label: self.label,
            kernel: k.clone(),
            globals: globals.clone(),
            pairs,
            hoisted,
            cutoff: t.cutoff,
        }))
    }
}

// ---------------------------------------------------------------------------
// Contribs
// ---------------------------------------------------------------------------

/// Common interface of the custom contribs.
pub trait CustomContrib: Contrib {
    fn kernel(&self) -> &Kernel;
    fn globals(&self) -> &Globals;
    /// `dE/d(global)`; 0 if the term does not depend on it.
    fn param_derivative(&self, name: &str, pos: &[f64]) -> f64;
}

/// Downcasts a contrib to a custom contrib.
pub fn as_custom(c: &dyn Contrib) -> Option<&dyn CustomContrib> {
    let any = c.as_any()?;
    if let Some(x) = any.downcast_ref::<CustomBondedContrib>() {
        return Some(x);
    }
    if let Some(x) = any.downcast_ref::<CustomNonbondedContrib>() {
        return Some(x);
    }
    None
}

fn load_globals(k: &Kernel, globals: &Globals, regs: &mut [f64]) {
    let off = k.global_offset();
    for (s, &gi) in k.global_indices().iter().enumerate() {
        regs[off + s] = globals.value(gi);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondedKind {
    Bond,
    Angle,
    Torsion,
    External,
}

impl BondedKind {
    fn arity(self) -> usize {
        match self {
            BondedKind::Bond => 2,
            BondedKind::Angle => 3,
            BondedKind::Torsion => 4,
            BondedKind::External => 1,
        }
    }
}

/// Bond / angle / torsion / external custom term.
#[derive(Debug, Clone)]
pub struct CustomBondedContrib {
    label: &'static str,
    kind: BondedKind,
    kernel: Arc<Kernel>,
    globals: Globals,
    atoms: Vec<usize>,
    hoisted: Vec<f64>,
}

struct AngleGeom {
    r0: [f64; 3],
    r1: [f64; 3],
    l0: f64,
    l1: f64,
    theta: f64,
}

fn angle_geom(pos: &[f64], i: usize, j: usize, k: usize) -> AngleGeom {
    let r0 = sub(p3(pos, i), p3(pos, j));
    let r1 = sub(p3(pos, k), p3(pos, j));
    let l0 = dot(r0, r0).max(1e-5);
    let l1 = dot(r1, r1).max(1e-5);
    let c = (dot(r0, r1) / (l0 * l1).sqrt()).clamp(-1.0, 1.0);
    AngleGeom { r0, r1, l0, l1, theta: c.acos() }
}

impl CustomBondedContrib {
    pub fn kind(&self) -> BondedKind {
        self.kind
    }
    pub fn num_terms(&self) -> usize {
        self.atoms.len() / self.kind.arity()
    }

    /// Sets the geometric inputs of term `t` in `regs`.
    #[inline]
    fn set_geom(&self, pos: &[f64], a: &[usize], regs: &mut [f64]) {
        match self.kind {
            BondedKind::Bond => regs[0] = norm(sub(p3(pos, a[0]), p3(pos, a[1]))),
            BondedKind::Angle => regs[0] = angle_geom(pos, a[0], a[1], a[2]).theta,
            BondedKind::Torsion => regs[0] = compute_dihedral(pos, a[0], a[1], a[2], a[3]).dihedral,
            BondedKind::External => regs[..3].copy_from_slice(&p3(pos, a[0])),
        }
    }

    /// Runs `prog` for every term, calling `f(term, regs)` after each run.
    fn each(&self, pos: &[f64], prog: &Program<f64>, mut f: impl FnMut(usize, &[f64])) {
        let k = &self.kernel;
        let nh = k.n_hoisted();
        let ar = self.kind.arity();
        let hoff = k.hoist_offset();
        let mut regs = prog.init_regs();
        load_globals(k, &self.globals, &mut regs);
        for t in 0..self.num_terms() {
            let a = &self.atoms[t * ar..(t + 1) * ar];
            self.set_geom(pos, a, &mut regs);
            regs[hoff..hoff + nh].copy_from_slice(&self.hoisted[t * nh..(t + 1) * nh]);
            prog.run(&mut regs);
            f(t, &regs);
        }
    }
}

impl Contrib for CustomBondedContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        let prog = &self.kernel.e;
        let o = prog.output_reg(0);
        let mut e = 0.0;
        self.each(pos, prog, |_, r| e += r[o]);
        e
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let prog = &self.kernel.eg;
        let ar = self.kind.arity();
        let o1 = prog.output_reg(1);
        let (o2, o3) =
            if self.kind == BondedKind::External { (prog.output_reg(2), prog.output_reg(3)) } else { (0, 0) };
        self.each(pos, prog, |t, r| {
            let a = &self.atoms[t * ar..(t + 1) * ar];
            let de = r[o1];
            match self.kind {
                BondedKind::Bond => {
                    let d = sub(p3(pos, a[0]), p3(pos, a[1]));
                    let g = scale(d, de / norm(d).max(1e-8));
                    add_grad(grad, a[0], g);
                    add_grad(grad, a[1], scale(g, -1.0));
                }
                BondedKind::Angle => {
                    let ag = angle_geom(pos, a[0], a[1], a[2]);
                    let rp = cross(ag.r1, ag.r0);
                    let pre = de / norm(rp).max(1e-5);
                    let d0 = scale(cross(ag.r0, rp), -pre / ag.l0);
                    let d2 = scale(cross(ag.r1, rp), pre / ag.l1);
                    let d1 = scale([d0[0] + d2[0], d0[1] + d2[1], d0[2] + d2[2]], -1.0);
                    add_grad(grad, a[0], d0);
                    add_grad(grad, a[1], d1);
                    add_grad(grad, a[2], d2);
                }
                BondedKind::Torsion => {
                    let idx = [a[0], a[1], a[2], a[3]];
                    let dd = compute_dihedral(pos, idx[0], idx[1], idx[2], idx[3]);
                    torsion_grad(pos, idx, &dd, de, grad);
                }
                BondedKind::External => add_grad(grad, a[0], [de, r[o2], r[o3]]),
            }
        });
    }
    fn label(&self) -> &'static str {
        self.label
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

impl CustomContrib for CustomBondedContrib {
    fn kernel(&self) -> &Kernel {
        &self.kernel
    }
    fn globals(&self) -> &Globals {
        &self.globals
    }
    fn param_derivative(&self, name: &str, pos: &[f64]) -> f64 {
        let Some(s) = self.kernel.global_local_index(name) else { return 0.0 };
        let prog = &self.kernel.dglobal;
        let o = prog.output_reg(s);
        let mut d = 0.0;
        self.each(pos, prog, |_, r| d += r[o]);
        d
    }
}

/// Pairwise custom term over a precomputed pair list.
#[derive(Debug, Clone)]
pub struct CustomNonbondedContrib {
    label: &'static str,
    kernel: Arc<Kernel>,
    globals: Globals,
    pairs: Vec<[u32; 2]>,
    /// `pairs.len() * n_hoisted` per-pair parameter values.
    hoisted: Vec<f64>,
    cutoff: Option<Cutoff>,
}

const LANES: usize = 16;
const PAR_PAIRS: usize = 32_768;
const PAR_CHUNK: usize = 8_192;

/// Switch value and derivative.
#[inline]
fn switch(c: &Cutoff, r: f64) -> (f64, f64) {
    match c.switch_from {
        Some(rs) if r > rs => {
            let w = c.cutoff - rs;
            let t = (r - rs) / w;
            let t2 = t * t;
            let s = 1.0 + t2 * t * (-10.0 + t * (15.0 - 6.0 * t));
            let ds = t2 * (-30.0 + t * (60.0 - 30.0 * t)) / w;
            (s, ds)
        }
        _ => (1.0, 0.0),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Energy,
    Grad,
    Param(usize),
}

impl CustomNonbondedContrib {
    pub fn num_pairs(&self) -> usize {
        self.pairs.len()
    }
    pub fn pairs(&self) -> &[[u32; 2]] {
        &self.pairs
    }

    /// Evaluates pairs `range`; returns the energy (or the parameter
    /// derivative) and adds the gradient in `Mode::Grad`.
    fn eval_range(&self, pos: &[f64], range: std::ops::Range<usize>, mode: Mode, mut grad: Option<&mut [f64]>) -> f64 {
        let k = &self.kernel;
        let prog = match mode {
            Mode::Energy => &k.e,
            Mode::Grad => &k.eg,
            Mode::Param(_) => &k.dglobal,
        };
        let out0 = prog.output_reg(match mode {
            Mode::Param(s) => s,
            _ => 0,
        });
        let out1 = if mode == Mode::Grad { prog.output_reg(1) } else { out0 };
        let nh = k.n_hoisted();
        let hoff = k.hoist_offset();
        let mut regs = prog.init_lanes::<LANES>();
        let goff = k.global_offset();
        for (s, &gi) in k.global_indices().iter().enumerate() {
            regs[goff + s] = [self.globals.value(gi); LANES];
        }
        let rc = self.cutoff.map(|c| c.cutoff).unwrap_or(f64::INFINITY);
        let mut lane_pair = [0usize; LANES];
        let mut lane_d = [[0.0; 3]; LANES];
        let mut n = 0;
        let mut total = 0.0;
        let flush = |regs: &mut Vec<[f64; LANES]>,
                     n: usize,
                     lane_pair: &[usize; LANES],
                     lane_d: &[[f64; 3]; LANES],
                     grad: &mut Option<&mut [f64]>| {
            prog.run_lanes(regs);
            let mut acc = 0.0;
            for l in 0..n {
                let r = regs[0][l];
                let v = regs[out0][l];
                let (s, ds) = match &self.cutoff {
                    Some(c) => switch(c, r),
                    None => (1.0, 0.0),
                };
                acc += v * s;
                if let Some(g) = grad.as_deref_mut() {
                    let de = regs[out1][l] * s + v * ds;
                    let [i, j] = self.pairs[lane_pair[l]];
                    let f = scale(lane_d[l], de / r.max(1e-8));
                    add_grad(g, i as usize, f);
                    add_grad(g, j as usize, scale(f, -1.0));
                }
            }
            acc
        };
        for p in range {
            let [i, j] = self.pairs[p];
            let d = sub(p3(pos, i as usize), p3(pos, j as usize));
            let r = norm(d);
            if r >= rc {
                continue;
            }
            regs[0][n] = r;
            for h in 0..nh {
                regs[hoff + h][n] = self.hoisted[p * nh + h];
            }
            lane_pair[n] = p;
            lane_d[n] = d;
            n += 1;
            if n == LANES {
                total += flush(&mut regs, n, &lane_pair, &lane_d, &mut grad);
                n = 0;
            }
        }
        if n > 0 {
            total += flush(&mut regs, n, &lane_pair, &lane_d, &mut grad);
        }
        total
    }

    fn eval_all(&self, pos: &[f64], mode: Mode) -> f64 {
        let np = self.pairs.len();
        if np < PAR_PAIRS {
            return self.eval_range(pos, 0..np, mode, None);
        }
        (0..np.div_ceil(PAR_CHUNK))
            .into_par_iter()
            .map(|c| self.eval_range(pos, c * PAR_CHUNK..((c + 1) * PAR_CHUNK).min(np), mode, None))
            .sum()
    }
}

impl Contrib for CustomNonbondedContrib {
    fn energy(&self, pos: &[f64]) -> f64 {
        self.eval_all(pos, Mode::Energy)
    }
    fn grad(&self, pos: &[f64], grad: &mut [f64]) {
        let np = self.pairs.len();
        if np < PAR_PAIRS {
            self.eval_range(pos, 0..np, Mode::Grad, Some(grad));
            return;
        }
        let n = pos.len();
        let g = (0..np.div_ceil(PAR_CHUNK))
            .into_par_iter()
            .fold(
                || vec![0.0; n],
                |mut g, c| {
                    self.eval_range(pos, c * PAR_CHUNK..((c + 1) * PAR_CHUNK).min(np), Mode::Grad, Some(&mut g));
                    g
                },
            )
            .reduce(
                || vec![0.0; n],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += *y);
                    a
                },
            );
        grad.iter_mut().zip(&g).for_each(|(x, y)| *x += *y);
    }
    fn label(&self) -> &'static str {
        self.label
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

impl CustomContrib for CustomNonbondedContrib {
    fn kernel(&self) -> &Kernel {
        &self.kernel
    }
    fn globals(&self) -> &Globals {
        &self.globals
    }
    fn param_derivative(&self, name: &str, pos: &[f64]) -> f64 {
        match self.kernel.global_local_index(name) {
            Some(s) => self.eval_all(pos, Mode::Param(s)),
            None => 0.0,
        }
    }
}
