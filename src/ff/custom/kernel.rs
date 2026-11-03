//! Lowering of parsed scripts into the IR, global parameters, and compiled
//! energy kernels (energy + symbolic derivatives + hoisted parameter
//! subexpressions).

use super::ir::{F1, F2, Graph, Id, Node};
use super::parse::{self, BinOp, ExprError, ExprKind, Script};
use super::program::Program;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Global parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct GlobalsInner {
    names: Vec<String>,
    values: Vec<AtomicU64>,
}

/// Named global parameters shared by custom terms (e.g. `lambda`).
///
/// Cloning is cheap and shares the values: every force field built from a
/// [`super::WithCustomTerms`] reads the current values at each evaluation,
/// so [`Globals::set`] changes already built force fields without
/// rebuilding them. Values are stored as `f64` bits in atomics, so setting
/// a value from another thread is safe (a concurrent evaluation sees either
/// the old or the new value of each global).
#[derive(Debug, Clone, Default)]
pub struct Globals {
    inner: Arc<GlobalsInner>,
}

impl Globals {
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns a new store with `name` added (or its value replaced). The new
    /// store does not share values with `self`.
    pub fn with(&self, name: &str, value: f64) -> Globals {
        let mut names = self.inner.names.clone();
        let mut values: Vec<AtomicU64> =
            self.inner.values.iter().map(|v| AtomicU64::new(v.load(Ordering::Relaxed))).collect();
        match names.iter().position(|n| n == name) {
            Some(i) => values[i] = AtomicU64::new(value.to_bits()),
            None => {
                names.push(name.to_string());
                values.push(AtomicU64::new(value.to_bits()));
            }
        }
        Globals { inner: Arc::new(GlobalsInner { names, values }) }
    }
    pub fn index(&self, name: &str) -> Option<usize> {
        self.inner.names.iter().position(|n| n == name)
    }
    pub fn names(&self) -> &[String] {
        &self.inner.names
    }
    pub fn len(&self) -> usize {
        self.inner.names.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.names.is_empty()
    }
    /// Current value of global `i`.
    #[inline]
    pub fn value(&self, i: usize) -> f64 {
        f64::from_bits(self.inner.values[i].load(Ordering::Relaxed))
    }
    pub fn get(&self, name: &str) -> Option<f64> {
        self.index(name).map(|i| self.value(i))
    }
    /// Sets an existing global. Errors if `name` was not declared.
    pub fn set(&self, name: &str, value: f64) -> crate::Result<()> {
        let i = self
            .index(name)
            .ok_or_else(|| crate::Error::InvalidArgument(format!("unknown global parameter '{name}'")))?;
        self.inner.values[i].store(value.to_bits(), Ordering::Relaxed);
        Ok(())
    }
    /// `true` if both handles share the same values.
    pub fn same_store(&self, other: &Globals) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

// ---------------------------------------------------------------------------
// Lowering
// ---------------------------------------------------------------------------

/// Reserved names that cannot be used as variables or definitions.
const BUILTIN_CONSTANTS: [(&str, f64); 1] = [("pi", std::f64::consts::PI)];

enum DefState {
    Todo,
    Busy,
    Done(Id),
}

/// Lowers statement `stmt` of `script` (and the definitions it uses) into
/// `g`. `lookup` maps free names to variable numbers.
pub(crate) fn lower_statement(
    script: &Script,
    stmt: usize,
    g: &mut Graph,
    lookup: &mut dyn FnMut(&str) -> Option<u32>,
) -> Result<Id, ExprError> {
    let src = &script.source;
    let mut defs: HashMap<&str, (usize, DefState)> = HashMap::new();
    for (k, st) in script.statements.iter().enumerate() {
        let Some((name, pos)) = &st.name else {
            if k > 0 {
                return Err(ExprError::new(
                    st.expr.pos,
                    "only the first statement may be a bare expression; the others must be definitions 'name = expression'",
                    src,
                ));
            }
            continue;
        };
        if defs.contains_key(name.as_str()) {
            return Err(ExprError::new(*pos, format!("'{name}' is defined twice"), src));
        }
        if BUILTIN_CONSTANTS.iter().any(|(c, _)| c == name)
            || F1::from_name(name).is_some()
            || F2::from_name(name).is_some()
            || name == "select"
        {
            return Err(ExprError::new(*pos, format!("'{name}' is a built-in name and cannot be redefined"), src));
        }
        if lookup(name).is_some() {
            return Err(ExprError::new(
                *pos,
                format!("definition of '{name}' shadows a variable or parameter of the same name"),
                src,
            ));
        }
        defs.insert(name.as_str(), (k, DefState::Todo));
    }
    let mut cx = Lower { script, g, lookup, defs };
    // The result statement itself is lowered directly (its name, if any, is
    // only a label).
    if let Some((name, _)) = &script.statements[stmt].name {
        cx.defs.get_mut(name.as_str()).unwrap().1 = DefState::Busy;
    }
    cx.expr(&script.statements[stmt].expr)
}

struct Lower<'a, 'b> {
    script: &'a Script,
    g: &'b mut Graph,
    lookup: &'b mut dyn FnMut(&str) -> Option<u32>,
    defs: HashMap<&'a str, (usize, DefState)>,
}

impl<'a> Lower<'a, '_> {
    fn err(&self, pos: usize, msg: impl Into<String>) -> ExprError {
        ExprError::new(pos, msg, &self.script.source)
    }

    fn name(&mut self, name: &'a str, pos: usize) -> Result<Id, ExprError> {
        if let Some((k, st)) = self.defs.get(name) {
            let k = *k;
            match st {
                DefState::Done(id) => return Ok(*id),
                DefState::Busy => return Err(self.err(pos, format!("cyclic definition: '{name}' depends on itself"))),
                DefState::Todo => {}
            }
            self.defs.get_mut(name).unwrap().1 = DefState::Busy;
            let script = self.script;
            let id = self.expr(&script.statements[k].expr)?;
            self.defs.get_mut(name).unwrap().1 = DefState::Done(id);
            return Ok(id);
        }
        if let Some(v) = (self.lookup)(name) {
            return Ok(self.g.var(v));
        }
        if let Some((_, c)) = BUILTIN_CONSTANTS.iter().find(|(c, _)| *c == name) {
            return Ok(self.g.constant(*c));
        }
        if F1::from_name(name).is_some() || F2::from_name(name).is_some() || name == "select" {
            return Err(self.err(pos, format!("function '{name}' used without arguments")));
        }
        Err(self.err(pos, format!("unknown variable '{name}'")))
    }

    fn expr(&mut self, e: &'a parse::Expr) -> Result<Id, ExprError> {
        Ok(match &e.kind {
            ExprKind::Num(v) => self.g.constant(*v),
            ExprKind::Var(n) => self.name(n, e.pos)?,
            ExprKind::Neg(a) => {
                let a = self.expr(a)?;
                self.g.neg(a)
            }
            ExprKind::Bin(op, a, b) => {
                let a = self.expr(a)?;
                let b = self.expr(b)?;
                let f = match op {
                    BinOp::Add => F2::Add,
                    BinOp::Sub => F2::Sub,
                    BinOp::Mul => F2::Mul,
                    BinOp::Div => F2::Div,
                    BinOp::Pow => F2::Pow,
                };
                self.g.f2(f, a, b)
            }
            ExprKind::Call(name, args) => {
                let arity = if F1::from_name(name).is_some() {
                    1
                } else if F2::from_name(name).is_some() {
                    2
                } else if name == "select" {
                    3
                } else {
                    return Err(self.err(
                        e.pos,
                        format!(
                            "unknown function '{name}' (available: {}, atan2, min, max, pow, select)",
                            F1::NAMED.iter().map(|f| f.name()).collect::<Vec<_>>().join(", ")
                        ),
                    ));
                };
                if args.len() != arity {
                    return Err(
                        self.err(e.pos, format!("function '{name}' takes {arity} argument(s), {} given", args.len()))
                    );
                }
                let mut ids = Vec::with_capacity(arity);
                for a in args {
                    ids.push(self.expr(a)?);
                }
                if let Some(f) = F1::from_name(name) {
                    self.g.f1(f, ids[0])
                } else if let Some(f) = F2::from_name(name) {
                    self.g.f2(f, ids[0], ids[1])
                } else {
                    self.g.select(ids[0], ids[1], ids[2])
                }
            }
        })
    }
}

/// Lowers a script with a fixed list of variable names (no globals).
pub fn lower(script: &Script, vars: &[&str]) -> Result<(Graph, Id), ExprError> {
    let mut g = Graph::new();
    let mut lookup = |n: &str| vars.iter().position(|v| *v == n).map(|i| i as u32);
    let r = lower_statement(script, 0, &mut g, &mut lookup)?;
    Ok((g, r))
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// A compiled energy function `E(geometry; parameters; globals)`.
///
/// Inputs of the evaluation programs are laid out as
/// `[geometry (n_geom) | hoisted (n_hoisted) | globals (n_globals)]`.
/// *Hoisted* values are the maximal subexpressions that depend only on the
/// per-term parameters (e.g. `4*eps*sig^12`): they are computed once per
/// term at build time by [`Kernel::hoist`].
#[derive(Clone, Debug)]
pub struct Kernel {
    source: String,
    geom: Vec<String>,
    params: Vec<String>,
    /// Names and store indices of the globals the energy depends on.
    global_names: Vec<String>,
    global_index: Vec<usize>,
    n_hoisted: usize,
    /// params -> hoisted values.
    hoist: Program<f64>,
    /// -> [E, dE/dgeom...]
    pub(crate) eg: Program<f64>,
    /// -> [E]
    pub(crate) e: Program<f64>,
    /// -> [dE/dglobal...]
    pub(crate) dglobal: Program<f64>,
    /// Unhoisted graph over `[geom | params | globals]` with roots
    /// `[E, dE/dgeom..., dE/dglobal...]` (for code generation and printing).
    full: Graph,
    full_roots: Vec<Id>,
    /// Hoisted graph over `[geom | hoisted | globals]` with the same roots.
    main: Graph,
    main_roots: Vec<Id>,
}

const GEOM: u8 = 1;
const PARAM: u8 = 2;
const GLOBAL: u8 = 4;

impl Kernel {
    /// Compiles `source` as a function of the geometric variables `geom`,
    /// the per-term parameters `params` and any globals declared in
    /// `globals`. Other free names are errors.
    pub fn compile(source: &str, geom: &[&str], params: &[&str], globals: &Globals) -> Result<Kernel, ExprError> {
        let script = parse::parse(source)?;
        Self::compile_script(&script, geom, params, globals)
    }

    pub(crate) fn compile_script(
        script: &Script,
        geom: &[&str],
        params: &[&str],
        globals: &Globals,
    ) -> Result<Kernel, ExprError> {
        let ng = geom.len();
        let np = params.len();
        for (k, p) in params.iter().enumerate() {
            if geom.contains(p) || params[..k].contains(p) {
                return Err(ExprError::new(
                    0,
                    format!("parameter name '{p}' is used twice or clashes with a geometric variable"),
                    &script.source,
                ));
            }
        }
        let mut global_names: Vec<String> = Vec::new();
        let mut global_index: Vec<usize> = Vec::new();
        let mut g = Graph::new();
        let e = {
            let mut lookup = |n: &str| -> Option<u32> {
                if let Some(i) = geom.iter().position(|v| *v == n) {
                    return Some(i as u32);
                }
                if let Some(i) = params.iter().position(|v| *v == n) {
                    return Some((ng + i) as u32);
                }
                if let Some(i) = global_names.iter().position(|v| v == n) {
                    return Some((ng + np + i) as u32);
                }
                let gi = globals.index(n)?;
                global_names.push(n.to_string());
                global_index.push(gi);
                Some((ng + np + global_names.len() - 1) as u32)
            };
            lower_statement(script, 0, &mut g, &mut lookup)?
        };
        // Drop globals that only appeared in unused definitions.
        let used = g.variables(e);
        let keep: Vec<usize> = (0..global_names.len()).filter(|&i| used.contains(&((ng + np + i) as u32))).collect();
        let remap_globals = keep.len() != global_names.len();
        let global_names: Vec<String> = keep.iter().map(|&i| global_names[i].clone()).collect();
        let global_index: Vec<usize> = keep.iter().map(|&i| global_index[i]).collect();
        let nu = global_names.len();
        let (mut g, e) = if remap_globals {
            let (g2, r) = g.extract(&[e], &|_| false, |g2, _, n| match n {
                Node::Var(k) if k as usize >= ng + np => {
                    let old = k as usize - ng - np;
                    let new = keep.iter().position(|&x| x == old).unwrap();
                    Some(g2.var((ng + np + new) as u32))
                }
                _ => None,
            });
            (g2, r[0])
        } else {
            (g, e)
        };

        let mut roots = vec![e];
        for v in 0..ng {
            roots.push(g.derivative(e, v as u32));
        }
        for k in 0..nu {
            roots.push(g.derivative(e, (ng + np + k) as u32));
        }

        // Dependency classes.
        let live = g.reachable(&roots);
        let mut dep = vec![0u8; g.len()];
        for i in 0..g.len() {
            if !live[i] {
                continue;
            }
            dep[i] = match g.node(i as Id) {
                Node::Const(_) => 0,
                Node::Var(k) if (k as usize) < ng => GEOM,
                Node::Var(k) if (k as usize) < ng + np => PARAM,
                Node::Var(_) => GLOBAL,
                n => n.children().fold(0, |acc, c| acc | dep[c as usize]),
            };
        }
        // Frontier: parameter-only nodes used by a non-parameter-only node or
        // that are roots.
        let mut frontier = vec![false; g.len()];
        for &r in &roots {
            if dep[r as usize] == PARAM {
                frontier[r as usize] = true;
            }
        }
        for i in 0..g.len() {
            if live[i] && dep[i] & (GEOM | GLOBAL) != 0 {
                for c in g.node(i as Id).children() {
                    if dep[c as usize] == PARAM {
                        frontier[c as usize] = true;
                    }
                }
            }
        }
        let hoisted: Vec<Id> = (0..g.len()).filter(|&i| frontier[i]).map(|i| i as Id).collect();
        let nh = hoisted.len();
        let slot: HashMap<Id, usize> = hoisted.iter().enumerate().map(|(s, &id)| (id, s)).collect();

        // Hoist program: params -> hoisted values.
        let (hg, hroots) = g.extract(&hoisted, &|_| false, |g2, _, n| match n {
            Node::Var(k) => Some(g2.var(k - ng as u32)),
            _ => None,
        });
        let (hg, hroots) = hg.expand_powers(&hroots);
        let hoist = Program::compile(&hg, &hroots, np);

        // Main graph: [geom | hoisted | globals].
        let (main, main_roots) = g.extract(&roots, &|id| slot.contains_key(&id), |g2, id, n| {
            if let Some(&s) = slot.get(&id) {
                return Some(g2.var((ng + s) as u32));
            }
            match n {
                Node::Var(k) if (k as usize) < ng => Some(g2.var(k)),
                Node::Var(k) if (k as usize) >= ng + np => Some(g2.var((k as usize - np + nh) as u32)),
                Node::Var(k) => unreachable!("parameter {k} outside the hoisted frontier"),
                _ => None,
            }
        });
        let n_in = ng + nh + nu;
        let (gx, rx) = main.expand_powers(&main_roots);
        let eg = Program::compile(&gx, &rx[..1 + ng], n_in);
        let e_prog = Program::compile(&gx, &rx[..1], n_in);
        let dglobal = Program::compile(&gx, &rx[1 + ng..], n_in);
        let (full, full_roots) = g.extract(&roots, &|_| false, |_, _, _| None);
        Ok(Kernel {
            source: script.source.clone(),
            geom: geom.iter().map(|s| s.to_string()).collect(),
            params: params.iter().map(|s| s.to_string()).collect(),
            global_names,
            global_index,
            n_hoisted: nh,
            hoist,
            eg,
            e: e_prog,
            dglobal,
            full,
            full_roots,
            main,
            main_roots,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn geometry_names(&self) -> &[String] {
        &self.geom
    }
    pub fn param_names(&self) -> &[String] {
        &self.params
    }
    /// Globals the energy depends on.
    pub fn global_names(&self) -> &[String] {
        &self.global_names
    }
    /// Store indices (in the [`Globals`] used at compile time) of
    /// [`Self::global_names`].
    pub fn global_indices(&self) -> &[usize] {
        &self.global_index
    }
    pub fn n_geom(&self) -> usize {
        self.geom.len()
    }
    pub fn n_params(&self) -> usize {
        self.params.len()
    }
    pub fn n_hoisted(&self) -> usize {
        self.n_hoisted
    }
    pub fn n_globals(&self) -> usize {
        self.global_names.len()
    }
    /// Offset of the hoisted inputs.
    #[inline]
    pub fn hoist_offset(&self) -> usize {
        self.geom.len()
    }
    /// Offset of the global inputs.
    #[inline]
    pub fn global_offset(&self) -> usize {
        self.geom.len() + self.n_hoisted
    }
    /// Computes the hoisted values for one term from its parameters.
    pub fn hoist(&self, params: &[f64], out: &mut [f64]) {
        assert_eq!(params.len(), self.params.len(), "wrong number of parameters");
        self.hoist.eval(params, out);
    }
    /// Programs: energy+geometric gradient, energy only, global derivatives.
    pub fn energy_grad_program(&self) -> &Program<f64> {
        &self.eg
    }
    pub fn energy_program(&self) -> &Program<f64> {
        &self.e
    }
    pub fn global_derivative_program(&self) -> &Program<f64> {
        &self.dglobal
    }
    /// Local index of a global name.
    pub fn global_local_index(&self, name: &str) -> Option<usize> {
        self.global_names.iter().position(|n| n == name)
    }

    /// Unhoisted graph over `[geom | params | globals]` and its roots
    /// `[E, dE/dgeom..., dE/dglobal...]`.
    pub fn graph(&self) -> (&Graph, &[Id]) {
        (&self.full, &self.full_roots)
    }
    /// Hoisted graph over `[geom | hoisted | globals]` with the same roots.
    pub fn hoisted_graph(&self) -> (&Graph, &[Id]) {
        (&self.main, &self.main_roots)
    }
    fn full_name(&self, k: u32) -> String {
        let k = k as usize;
        let (ng, np) = (self.geom.len(), self.params.len());
        if k < ng {
            self.geom[k].clone()
        } else if k < ng + np {
            self.params[k - ng].clone()
        } else {
            self.global_names[k - ng - np].clone()
        }
    }
    /// The (simplified) energy expression.
    pub fn energy_source(&self) -> String {
        self.full.to_source(self.full_roots[0], &|k| self.full_name(k))
    }
    /// The (simplified) derivative with respect to geometric variable `i`.
    pub fn derivative_source(&self, i: usize) -> String {
        self.full.to_source(self.full_roots[1 + i], &|k| self.full_name(k))
    }
    /// Evaluates energy and geometric derivatives directly from parameters
    /// (slow reference path; used in tests).
    pub fn eval_reference(&self, geom: &[f64], params: &[f64], globals: &[f64]) -> Vec<f64> {
        let mut vars = geom.to_vec();
        vars.extend_from_slice(params);
        vars.extend_from_slice(globals);
        self.full_roots[..1 + self.geom.len()].iter().map(|&r| self.full.eval(r, &vars)).collect()
    }
}
