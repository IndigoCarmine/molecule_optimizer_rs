//! Expression IR: a hash-consed DAG (structurally equal subexpressions are
//! shared, so common-subexpression elimination is automatic) with constant
//! folding and algebraic simplification in the node constructors, plus
//! symbolic differentiation.
//!
//! Node ids are allocated in creation order and children are always created
//! before their parents, so ascending id order is a topological order. This
//! is what the bytecode compiler ([`super::program`]) and the WGSL emitter
//! ([`super::wgsl`]) rely on.

use super::math::{self, Real};
use std::collections::HashMap;

/// Node index in a [`Graph`].
pub type Id = u32;

/// Unary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F1 {
    Neg,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Acos,
    Asin,
    Atan,
    Abs,
    Step,
    Sign,
    Erf,
    Erfc,
    Cosh,
    Sinh,
    Tanh,
}

/// Binary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F2 {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Atan2,
    Min,
    Max,
}

impl F1 {
    /// All unary functions callable by name (everything except `Neg`).
    pub const NAMED: [F1; 17] = [
        F1::Sqrt,
        F1::Exp,
        F1::Log,
        F1::Sin,
        F1::Cos,
        F1::Tan,
        F1::Acos,
        F1::Asin,
        F1::Atan,
        F1::Abs,
        F1::Step,
        F1::Sign,
        F1::Erf,
        F1::Erfc,
        F1::Cosh,
        F1::Sinh,
        F1::Tanh,
    ];
    pub fn name(self) -> &'static str {
        match self {
            F1::Neg => "-",
            F1::Sqrt => "sqrt",
            F1::Exp => "exp",
            F1::Log => "log",
            F1::Sin => "sin",
            F1::Cos => "cos",
            F1::Tan => "tan",
            F1::Acos => "acos",
            F1::Asin => "asin",
            F1::Atan => "atan",
            F1::Abs => "abs",
            F1::Step => "step",
            F1::Sign => "sign",
            F1::Erf => "erf",
            F1::Erfc => "erfc",
            F1::Cosh => "cosh",
            F1::Sinh => "sinh",
            F1::Tanh => "tanh",
        }
    }
    pub fn from_name(s: &str) -> Option<F1> {
        F1::NAMED.iter().copied().find(|f| f.name() == s)
    }
    #[inline]
    pub fn apply<T: Real>(self, x: T) -> T {
        match self {
            F1::Neg => -x,
            F1::Sqrt => x.sqrt(),
            F1::Exp => x.exp(),
            F1::Log => x.ln(),
            F1::Sin => x.sin(),
            F1::Cos => x.cos(),
            F1::Tan => x.tan(),
            F1::Acos => x.acos(),
            F1::Asin => x.asin(),
            F1::Atan => x.atan(),
            F1::Abs => x.abs(),
            F1::Step => math::step(x),
            F1::Sign => math::sign(x),
            F1::Erf => x.erf(),
            F1::Erfc => x.erfc(),
            F1::Cosh => x.cosh(),
            F1::Sinh => x.sinh(),
            F1::Tanh => x.tanh(),
        }
    }
}

impl F2 {
    pub fn name(self) -> &'static str {
        match self {
            F2::Add => "+",
            F2::Sub => "-",
            F2::Mul => "*",
            F2::Div => "/",
            F2::Pow => "^",
            F2::Atan2 => "atan2",
            F2::Min => "min",
            F2::Max => "max",
        }
    }
    pub fn from_name(s: &str) -> Option<F2> {
        match s {
            "atan2" => Some(F2::Atan2),
            "min" => Some(F2::Min),
            "max" => Some(F2::Max),
            "pow" => Some(F2::Pow),
            _ => None,
        }
    }
    #[inline]
    pub fn apply<T: Real>(self, a: T, b: T) -> T {
        match self {
            F2::Add => a + b,
            F2::Sub => a - b,
            F2::Mul => a * b,
            F2::Div => a / b,
            F2::Pow => a.powf(b),
            F2::Atan2 => a.atan2(b),
            F2::Min => math::min(a, b),
            F2::Max => math::max(a, b),
        }
    }
    fn commutative(self) -> bool {
        matches!(self, F2::Add | F2::Mul | F2::Min | F2::Max)
    }
}

/// A DAG node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    /// Constant (bit pattern of an `f64`; `-0.0` is normalized to `0.0`).
    Const(u64),
    /// Input variable number `k`.
    Var(u32),
    Un(F1, Id),
    Bin(F2, Id, Id),
    /// `x^n` for an integer constant `n`.
    Powi(Id, i32),
    /// `select(c, a, b)`: `a` if `c != 0`, else `b`.
    Select(Id, Id, Id),
}

impl Node {
    /// Child ids.
    pub fn children(&self) -> impl Iterator<Item = Id> {
        let (a, b, c) = match *self {
            Node::Const(_) | Node::Var(_) => (None, None, None),
            Node::Un(_, a) | Node::Powi(a, _) => (Some(a), None, None),
            Node::Bin(_, a, b) => (Some(a), Some(b), None),
            Node::Select(c, a, b) => (Some(c), Some(a), Some(b)),
        };
        a.into_iter().chain(b).chain(c)
    }
}

/// Hash-consed expression DAG.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    map: HashMap<Node, Id>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn node(&self, id: Id) -> Node {
        self.nodes[id as usize]
    }
    /// Constant value of `id`, if it is a constant.
    pub fn const_value(&self, id: Id) -> Option<f64> {
        match self.nodes[id as usize] {
            Node::Const(b) => Some(f64::from_bits(b)),
            _ => None,
        }
    }
    fn is_const(&self, id: Id, v: f64) -> bool {
        self.const_value(id) == Some(v)
    }

    fn intern(&mut self, n: Node) -> Id {
        if let Some(&id) = self.map.get(&n) {
            return id;
        }
        let id = self.nodes.len() as Id;
        self.nodes.push(n);
        self.map.insert(n, id);
        id
    }

    pub fn constant(&mut self, v: f64) -> Id {
        let v = if v == 0.0 { 0.0 } else { v };
        self.intern(Node::Const(v.to_bits()))
    }
    pub fn var(&mut self, k: u32) -> Id {
        self.intern(Node::Var(k))
    }

    pub fn neg(&mut self, a: Id) -> Id {
        if let Some(v) = self.const_value(a) {
            return self.constant(-v);
        }
        match self.node(a) {
            Node::Un(F1::Neg, x) => x,
            Node::Bin(F2::Sub, x, y) => self.sub(y, x),
            _ => self.intern(Node::Un(F1::Neg, a)),
        }
    }

    pub fn add(&mut self, a: Id, b: Id) -> Id {
        if self.is_const(a, 0.0) {
            return b;
        }
        if self.is_const(b, 0.0) {
            return a;
        }
        if let Node::Un(F1::Neg, y) = self.node(b) {
            return self.sub(a, y);
        }
        if let Node::Un(F1::Neg, x) = self.node(a) {
            return self.sub(b, x);
        }
        if a == b {
            let two = self.constant(2.0);
            return self.mul(two, a);
        }
        // (x - y) + y = x
        if let Node::Bin(F2::Sub, x, y) = self.node(a)
            && y == b
        {
            return x;
        }
        if let Node::Bin(F2::Sub, x, y) = self.node(b)
            && y == a
        {
            return x;
        }
        self.bin_raw(F2::Add, a, b)
    }

    pub fn sub(&mut self, a: Id, b: Id) -> Id {
        if self.is_const(b, 0.0) {
            return a;
        }
        if self.is_const(a, 0.0) {
            return self.neg(b);
        }
        if a == b {
            return self.constant(0.0);
        }
        if let Node::Un(F1::Neg, y) = self.node(b) {
            return self.add(a, y);
        }
        // (x + y) - y = x, (x + y) - x = y
        if let Node::Bin(F2::Add, x, y) = self.node(a) {
            if y == b {
                return x;
            }
            if x == b {
                return y;
            }
        }
        self.bin_raw(F2::Sub, a, b)
    }

    pub fn mul(&mut self, a: Id, b: Id) -> Id {
        // Constants first.
        let (a, b) = if self.const_value(b).is_some() && self.const_value(a).is_none() { (b, a) } else { (a, b) };
        if let Some(ca) = self.const_value(a) {
            if ca == 0.0 {
                return a;
            }
            if ca == 1.0 {
                return b;
            }
            if ca == -1.0 {
                return self.neg(b);
            }
            // c1 * (c2 * x) -> (c1*c2) * x
            if let Node::Bin(F2::Mul, x, y) = self.node(b)
                && let Some(cx) = self.const_value(x)
            {
                let c = self.constant(ca * cx);
                return self.mul(c, y);
            }
        }
        if let Node::Un(F1::Neg, x) = self.node(a) {
            let m = self.mul(x, b);
            return self.neg(m);
        }
        if let Node::Un(F1::Neg, y) = self.node(b) {
            let m = self.mul(a, y);
            return self.neg(m);
        }
        // Powers of the same base.
        let (ba, ea) = self.as_power(a);
        let (bb, eb) = self.as_power(b);
        if ba == bb && self.const_value(ba).is_none() {
            return self.powi(ba, ea + eb);
        }
        self.bin_raw(F2::Mul, a, b)
    }

    fn as_power(&self, a: Id) -> (Id, i32) {
        match self.node(a) {
            Node::Powi(x, n) => (x, n),
            _ => (a, 1),
        }
    }

    pub fn div(&mut self, a: Id, b: Id) -> Id {
        if self.is_const(b, 1.0) {
            return a;
        }
        if self.is_const(a, 0.0) && self.const_value(b).is_none_or(|v| v != 0.0) {
            return a;
        }
        if let Some(cb) = self.const_value(b)
            && cb != 0.0
            && cb.is_finite()
        {
            // Division by a power of two is an exact multiplication.
            let r = 1.0 / cb;
            let (m, _) = frexp(r);
            if m.abs() == 0.5 {
                let c = self.constant(r);
                return self.mul(c, a);
            }
        }
        if let Node::Un(F1::Neg, x) = self.node(a) {
            let d = self.div(x, b);
            return self.neg(d);
        }
        if let Node::Un(F1::Neg, y) = self.node(b) {
            let d = self.div(a, y);
            return self.neg(d);
        }
        self.bin_raw(F2::Div, a, b)
    }

    pub fn powi(&mut self, a: Id, n: i32) -> Id {
        if n == 0 {
            return self.constant(1.0);
        }
        if n == 1 {
            return a;
        }
        if let Some(v) = self.const_value(a) {
            return self.constant(math::powi(v, n));
        }
        match self.node(a) {
            Node::Powi(x, m) if (m as i64 * n as i64).abs() <= i32::MAX as i64 => self.powi(x, m * n),
            Node::Un(F1::Neg, x) if n % 2 == 0 => self.powi(x, n),
            Node::Un(F1::Neg, x) => {
                let p = self.powi(x, n);
                self.neg(p)
            }
            _ => self.intern(Node::Powi(a, n)),
        }
    }

    pub fn pow(&mut self, a: Id, b: Id) -> Id {
        if let Some(e) = self.const_value(b) {
            if e == e.trunc() && e.abs() <= 1024.0 {
                return self.powi(a, e as i32);
            }
            if e == 0.5 {
                return self.f1(F1::Sqrt, a);
            }
        }
        self.bin_raw(F2::Pow, a, b)
    }

    pub fn f1(&mut self, f: F1, a: Id) -> Id {
        if f == F1::Neg {
            return self.neg(a);
        }
        if let Some(v) = self.const_value(a) {
            return self.constant(f.apply(v));
        }
        match (f, self.node(a)) {
            // Idempotent functions.
            (F1::Abs | F1::Step | F1::Sign, Node::Un(g, _)) if g == f => a,
            (F1::Abs, Node::Un(F1::Neg, x)) => self.f1(F1::Abs, x),
            _ => self.intern(Node::Un(f, a)),
        }
    }

    /// Binary operation dispatch with simplification.
    pub fn f2(&mut self, f: F2, a: Id, b: Id) -> Id {
        match f {
            F2::Add => self.add(a, b),
            F2::Sub => self.sub(a, b),
            F2::Mul => self.mul(a, b),
            F2::Div => self.div(a, b),
            F2::Pow => self.pow(a, b),
            _ => {
                if (f == F2::Min || f == F2::Max) && a == b {
                    return a;
                }
                self.bin_raw(f, a, b)
            }
        }
    }

    fn bin_raw(&mut self, f: F2, a: Id, b: Id) -> Id {
        if let (Some(x), Some(y)) = (self.const_value(a), self.const_value(b)) {
            return self.constant(f.apply(x, y));
        }
        let (a, b) = if f.commutative() && self.order_key(b) < self.order_key(a) { (b, a) } else { (a, b) };
        self.intern(Node::Bin(f, a, b))
    }

    /// Canonical operand order for commutative operations: constants first,
    /// then by id.
    fn order_key(&self, id: Id) -> (u8, Id) {
        (u8::from(self.const_value(id).is_none()), id)
    }

    pub fn select(&mut self, c: Id, a: Id, b: Id) -> Id {
        if let Some(v) = self.const_value(c) {
            return if v != 0.0 { a } else { b };
        }
        if a == b {
            return a;
        }
        self.intern(Node::Select(c, a, b))
    }

    /// Rebuilds `node` with new children (through the simplifying constructors).
    fn rebuild_node(&mut self, n: Node, map: &dyn Fn(Id) -> Id) -> Id {
        match n {
            Node::Const(b) => self.constant(f64::from_bits(b)),
            Node::Var(k) => self.var(k),
            Node::Un(f, a) => self.f1(f, map(a)),
            Node::Bin(f, a, b) => self.f2(f, map(a), map(b)),
            Node::Powi(a, k) => self.powi(map(a), k),
            Node::Select(c, a, b) => self.select(map(c), map(a), map(b)),
        }
    }

    /// Marks the nodes reachable from `roots`.
    pub fn reachable(&self, roots: &[Id]) -> Vec<bool> {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack: Vec<Id> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if std::mem::replace(&mut seen[id as usize], true) {
                continue;
            }
            stack.extend(self.nodes[id as usize].children());
        }
        seen
    }

    /// Copies the subgraph of `roots` into a new graph. `subst(id)` may
    /// replace a node by a node of the new graph (e.g. a variable); other
    /// nodes are rebuilt (and re-simplified) from their mapped children.
    ///
    /// Nodes for which `cut(id)` is true are not descended into (their
    /// children are only copied if reachable otherwise); `subst` must map them.
    pub fn extract(
        &self,
        roots: &[Id],
        cut: &dyn Fn(Id) -> bool,
        mut subst: impl FnMut(&mut Graph, Id, Node) -> Option<Id>,
    ) -> (Graph, Vec<Id>) {
        let mut live = vec![false; self.nodes.len()];
        let mut stack: Vec<Id> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if std::mem::replace(&mut live[id as usize], true) {
                continue;
            }
            if !cut(id) {
                stack.extend(self.nodes[id as usize].children());
            }
        }
        let mut g = Graph::new();
        let mut map = vec![Id::MAX; self.nodes.len()];
        for (i, &l) in live.iter().enumerate() {
            if !l {
                continue;
            }
            let n = self.nodes[i];
            let new = match subst(&mut g, i as Id, n) {
                Some(x) => x,
                None => {
                    assert!(!cut(i as Id), "cut node {i} was not substituted");
                    g.rebuild_node(n, &|c| map[c as usize])
                }
            };
            map[i] = new;
        }
        let r = roots.iter().map(|&r| map[r as usize]).collect();
        (g, r)
    }

    /// Lowers integer powers with `|n| > 3` into shared square-and-multiply
    /// chains (`x^12 = ((x^2)^2 * x^2)^2`-style, so `x^12`, `x^11`, `x^6`,
    /// `x^5` of the same base share their intermediate powers). Used before
    /// bytecode compilation; the result is not re-simplified.
    pub fn expand_powers(&self, roots: &[Id]) -> (Graph, Vec<Id>) {
        fn chain(g: &mut Graph, x: Id, n: u32) -> Id {
            match n {
                1 => x,
                2 | 3 => g.intern(Node::Powi(x, n as i32)),
                _ => {
                    let h = chain(g, x, n / 2);
                    let sq = g.intern(Node::Powi(h, 2));
                    if n % 2 == 1 { g.intern(Node::Bin(F2::Mul, sq, x)) } else { sq }
                }
            }
        }
        let live = self.reachable(roots);
        let mut g = Graph::new();
        let mut map = vec![Id::MAX; self.nodes.len()];
        for i in 0..self.nodes.len() {
            if !live[i] {
                continue;
            }
            let m = |c: Id| map[c as usize];
            map[i] = match self.nodes[i] {
                Node::Powi(a, n) if n.unsigned_abs() > 3 => {
                    let p = chain(&mut g, m(a), n.unsigned_abs());
                    if n < 0 {
                        let one = g.constant(1.0);
                        g.intern(Node::Bin(F2::Div, one, p))
                    } else {
                        p
                    }
                }
                Node::Const(b) => g.intern(Node::Const(b)),
                Node::Var(k) => g.intern(Node::Var(k)),
                Node::Un(f, a) => g.intern(Node::Un(f, m(a))),
                Node::Bin(f, a, b) => g.intern(Node::Bin(f, m(a), m(b))),
                Node::Powi(a, n) => g.intern(Node::Powi(m(a), n)),
                Node::Select(c, a, b) => g.intern(Node::Select(m(c), m(a), m(b))),
            };
        }
        let r = roots.iter().map(|&r| map[r as usize]).collect();
        (g, r)
    }

    /// Symbolic derivative of `root` with respect to variable `var`.
    pub fn derivative(&mut self, root: Id, var: u32) -> Id {
        let mut memo: HashMap<Id, Id> = HashMap::new();
        self.d(root, var, &mut memo)
    }

    fn d(&mut self, id: Id, var: u32, memo: &mut HashMap<Id, Id>) -> Id {
        if let Some(&r) = memo.get(&id) {
            return r;
        }
        let r = match self.node(id) {
            Node::Const(_) => self.constant(0.0),
            Node::Var(k) => self.constant(if k == var { 1.0 } else { 0.0 }),
            Node::Un(f, a) => {
                let da = self.d(a, var, memo);
                if self.is_const(da, 0.0) {
                    da
                } else {
                    let outer = self.d_unary(f, a, id);
                    self.mul(outer, da)
                }
            }
            Node::Powi(a, n) => {
                let da = self.d(a, var, memo);
                if self.is_const(da, 0.0) {
                    da
                } else {
                    let p = self.powi(a, n - 1);
                    let c = self.constant(n as f64);
                    let t = self.mul(c, p);
                    self.mul(t, da)
                }
            }
            Node::Bin(f, a, b) => {
                let da = self.d(a, var, memo);
                let db = self.d(b, var, memo);
                self.d_binary(f, a, b, id, da, db)
            }
            Node::Select(c, a, b) => {
                let da = self.d(a, var, memo);
                let db = self.d(b, var, memo);
                self.select(c, da, db)
            }
        };
        memo.insert(id, r);
        r
    }

    /// d f(a) / d a, where `this = f(a)`.
    fn d_unary(&mut self, f: F1, a: Id, this: Id) -> Id {
        match f {
            F1::Neg => self.constant(-1.0),
            F1::Sqrt => {
                let two = self.constant(2.0);
                let t = self.mul(two, this);
                let one = self.constant(1.0);
                self.div(one, t)
            }
            F1::Exp => this,
            F1::Log => {
                let one = self.constant(1.0);
                self.div(one, a)
            }
            F1::Sin => self.f1(F1::Cos, a),
            F1::Cos => {
                let s = self.f1(F1::Sin, a);
                self.neg(s)
            }
            F1::Tan => {
                let one = self.constant(1.0);
                let t2 = self.powi(this, 2);
                self.add(one, t2)
            }
            F1::Acos | F1::Asin => {
                let one = self.constant(1.0);
                let a2 = self.powi(a, 2);
                let s = self.sub(one, a2);
                let q = self.f1(F1::Sqrt, s);
                let r = self.div(one, q);
                if f == F1::Acos { self.neg(r) } else { r }
            }
            F1::Atan => {
                let one = self.constant(1.0);
                let a2 = self.powi(a, 2);
                let s = self.add(one, a2);
                self.div(one, s)
            }
            F1::Abs => self.f1(F1::Sign, a),
            F1::Step | F1::Sign => self.constant(0.0),
            F1::Erf | F1::Erfc => {
                let a2 = self.powi(a, 2);
                let na2 = self.neg(a2);
                let e = self.f1(F1::Exp, na2);
                let c = self.constant(if f == F1::Erf { 1.0 } else { -1.0 } * std::f64::consts::FRAC_2_SQRT_PI);
                self.mul(c, e)
            }
            F1::Cosh => self.f1(F1::Sinh, a),
            F1::Sinh => self.f1(F1::Cosh, a),
            F1::Tanh => {
                let one = self.constant(1.0);
                let t2 = self.powi(this, 2);
                self.sub(one, t2)
            }
        }
    }

    fn d_binary(&mut self, f: F2, a: Id, b: Id, this: Id, da: Id, db: Id) -> Id {
        let za = self.is_const(da, 0.0);
        let zb = self.is_const(db, 0.0);
        if za && zb {
            return da;
        }
        match f {
            F2::Add => self.add(da, db),
            F2::Sub => self.sub(da, db),
            F2::Mul => {
                let x = self.mul(da, b);
                let y = self.mul(a, db);
                self.add(x, y)
            }
            F2::Div => {
                // (da - (a/b) db) / b
                let t = self.mul(this, db);
                let n = self.sub(da, t);
                self.div(n, b)
            }
            F2::Pow => {
                // a^b (db ln a + b da / a)
                let mut s = self.constant(0.0);
                if !zb {
                    let l = self.f1(F1::Log, a);
                    s = self.mul(db, l);
                }
                if !za {
                    let q = self.div(da, a);
                    let t = self.mul(b, q);
                    s = self.add(s, t);
                }
                self.mul(this, s)
            }
            F2::Atan2 => {
                // (b da - a db) / (a^2 + b^2) for atan2(a, b)
                let x = self.mul(b, da);
                let y = self.mul(a, db);
                let n = self.sub(x, y);
                let a2 = self.powi(a, 2);
                let b2 = self.powi(b, 2);
                let den = self.add(a2, b2);
                self.div(n, den)
            }
            F2::Min | F2::Max => {
                // min: da where a <= b; max: da where a >= b.
                let diff = if f == F2::Min { self.sub(b, a) } else { self.sub(a, b) };
                let c = self.f1(F1::Step, diff);
                self.select(c, da, db)
            }
        }
    }

    /// Evaluates `root` with variable values `vars` (slow; for tests and
    /// constant evaluation).
    pub fn eval(&self, root: Id, vars: &[f64]) -> f64 {
        let live = self.reachable(&[root]);
        let mut v = vec![0.0; self.nodes.len()];
        for i in 0..=root as usize {
            if !live[i] {
                continue;
            }
            v[i] = match self.nodes[i] {
                Node::Const(b) => f64::from_bits(b),
                Node::Var(k) => vars[k as usize],
                Node::Un(f, a) => f.apply(v[a as usize]),
                Node::Bin(f, a, b) => f.apply(v[a as usize], v[b as usize]),
                Node::Powi(a, n) => math::powi(v[a as usize], n),
                Node::Select(c, a, b) => math::select(v[c as usize], v[a as usize], v[b as usize]),
            };
        }
        v[root as usize]
    }

    /// Variables `root` depends on.
    pub fn variables(&self, root: Id) -> Vec<u32> {
        let live = self.reachable(&[root]);
        let mut out: Vec<u32> = live
            .iter()
            .enumerate()
            .filter(|(_, l)| **l)
            .filter_map(|(i, _)| match self.nodes[i] {
                Node::Var(k) => Some(k),
                _ => None,
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Number of operation (non-leaf) nodes reachable from `roots`.
    pub fn op_count(&self, roots: &[Id]) -> usize {
        let live = self.reachable(roots);
        live.iter().enumerate().filter(|(i, l)| **l && !matches!(self.nodes[*i], Node::Const(_) | Node::Var(_))).count()
    }

    /// Prints `root` as an expression in the input language (the DAG is
    /// expanded into a tree). `name(k)` gives variable names.
    pub fn to_source(&self, root: Id, name: &dyn Fn(u32) -> String) -> String {
        let mut s = String::new();
        self.write(&mut s, root, name);
        s
    }

    fn prec(&self, id: Id) -> u8 {
        match self.node(id) {
            Node::Const(b) if f64::from_bits(b) < 0.0 => 3,
            Node::Const(_) | Node::Var(_) | Node::Select(..) => 5,
            Node::Un(F1::Neg, _) => 3,
            Node::Un(..) => 5,
            Node::Powi(..) => 4,
            Node::Bin(F2::Add | F2::Sub, ..) => 1,
            Node::Bin(F2::Mul | F2::Div, ..) => 2,
            Node::Bin(F2::Pow, ..) => 4,
            Node::Bin(..) => 5,
        }
    }

    fn write_child(&self, s: &mut String, id: Id, paren: bool, name: &dyn Fn(u32) -> String) {
        if paren {
            s.push('(');
        }
        self.write(s, id, name);
        if paren {
            s.push(')');
        }
    }

    fn write(&self, s: &mut String, id: Id, name: &dyn Fn(u32) -> String) {
        match self.node(id) {
            Node::Const(b) => {
                let v = f64::from_bits(b);
                if v.is_finite() {
                    if v < 0.0 {
                        s.push('-');
                    }
                    s.push_str(&super::parse::format_number(v.abs()));
                } else if v.is_nan() {
                    s.push_str("(0/0)");
                } else {
                    s.push_str(if v > 0.0 { "(1/0)" } else { "(-1/0)" });
                }
            }
            Node::Var(k) => s.push_str(&name(k)),
            Node::Un(F1::Neg, a) => {
                s.push('-');
                self.write_child(s, a, self.prec(a) < 3, name);
            }
            Node::Un(f, a) => {
                s.push_str(f.name());
                s.push('(');
                self.write(s, a, name);
                s.push(')');
            }
            Node::Powi(a, n) => {
                self.write_child(s, a, self.prec(a) <= 4, name);
                s.push('^');
                if n < 0 {
                    s.push_str(&format!("({n})"));
                } else {
                    s.push_str(&n.to_string());
                }
            }
            Node::Bin(f @ (F2::Atan2 | F2::Min | F2::Max), a, b) => {
                s.push_str(f.name());
                s.push('(');
                self.write(s, a, name);
                s.push_str(", ");
                self.write(s, b, name);
                s.push(')');
            }
            Node::Bin(F2::Pow, a, b) => {
                self.write_child(s, a, self.prec(a) <= 4, name);
                s.push('^');
                self.write_child(s, b, self.prec(b) < 4, name);
            }
            Node::Bin(f, a, b) => {
                let p = self.prec(id);
                self.write_child(s, a, self.prec(a) < p, name);
                if p == 1 {
                    s.push(' ');
                    s.push_str(f.name());
                    s.push(' ');
                } else {
                    s.push_str(f.name());
                }
                self.write_child(s, b, self.prec(b) <= p, name);
            }
            Node::Select(c, a, b) => {
                s.push_str("select(");
                self.write(s, c, name);
                s.push_str(", ");
                self.write(s, a, name);
                s.push_str(", ");
                self.write(s, b, name);
                s.push(')');
            }
        }
    }
}

/// `(m, e)` with `x = m 2^e`, `0.5 <= |m| < 1` (finite non-zero `x`).
fn frexp(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        // Subnormal: scale up first.
        let (m, e) = frexp(x * 2f64.powi(64));
        return (m, e - 64);
    }
    let m = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (m, exp - 1022)
}
