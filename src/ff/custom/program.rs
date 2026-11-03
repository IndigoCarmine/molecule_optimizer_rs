//! Register bytecode compiled from the IR.
//!
//! Register file layout: `[inputs | constants | temporaries]`. Inputs and
//! constants are never overwritten by instructions, so a register file can
//! be initialized once ([`Program::init_regs`]) and reused for any number of
//! evaluations: set the inputs, [`Program::run`], read the outputs. No
//! allocation happens per evaluation.
//!
//! Temporaries are allocated by a linear scan over the topologically ordered
//! DAG with reuse after the last use; an instruction's destination never
//! aliases its operands.
//!
//! [`Program::run_lanes`] evaluates `L` independent inputs at once
//! (structure-of-arrays registers `[T; L]`): one instruction dispatch per `L`
//! evaluations, and the per-lane inner loops auto-vectorize for the
//! arithmetic operations.

use super::ir::{F1, F2, Graph, Id, Node};
use super::math::{self, Real};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Code {
    Un(F1),
    Bin(F2),
    Powi,
    Select,
}

/// One instruction: `regs[d] = op(regs[a], regs[b], regs[c])`.
#[derive(Clone, Copy, Debug)]
pub struct Instr {
    pub code: Code,
    pub d: u16,
    pub a: u16,
    pub b: u16,
    pub c: u16,
    /// Integer exponent of `Powi`.
    pub n: i32,
}

/// Compiled straight-line program.
#[derive(Clone, Debug)]
pub struct Program<T: Real> {
    n_inputs: usize,
    n_regs: usize,
    consts: Vec<(u16, T)>,
    code: Vec<Instr>,
    outputs: Vec<u16>,
}

impl Program<f64> {
    /// Compiles `roots` of `g`. Variables `Var(k)` read input `k < n_inputs`.
    pub fn compile(g: &Graph, roots: &[Id], n_inputs: usize) -> Program<f64> {
        let n = g.len();
        let live = g.reachable(roots);
        // Last use (node id of the last consumer); roots live to the end.
        let mut last_use = vec![0u32; n];
        for i in 0..n {
            if live[i] {
                for c in g.node(i as Id).children() {
                    last_use[c as usize] = last_use[c as usize].max(i as u32);
                }
            }
        }
        for &r in roots {
            last_use[r as usize] = u32::MAX;
        }
        let mut reg = vec![u16::MAX; n];
        let mut consts = Vec::new();
        let mut next = n_inputs;
        for i in 0..n {
            if !live[i] {
                continue;
            }
            match g.node(i as Id) {
                Node::Var(k) => {
                    assert!((k as usize) < n_inputs, "variable {k} out of range ({n_inputs} inputs)");
                    reg[i] = k as u16;
                }
                Node::Const(b) => {
                    reg[i] = next as u16;
                    consts.push((next as u16, f64::from_bits(b)));
                    next += 1;
                }
                _ => {}
            }
        }
        let first_temp = next;
        let mut free: Vec<u16> = Vec::new();
        let mut n_regs = next;
        let mut code = Vec::new();
        for i in 0..n {
            if !live[i] {
                continue;
            }
            let node = g.node(i as Id);
            let (code_, a, b, c, k) = match node {
                Node::Var(_) | Node::Const(_) => continue,
                Node::Un(f, a) => (Code::Un(f), a, a, a, 0),
                Node::Bin(f, a, b) => (Code::Bin(f), a, b, b, 0),
                Node::Powi(a, k) => (Code::Powi, a, a, a, k),
                Node::Select(c, a, b) => (Code::Select, c, a, b, 0),
            };
            let d = free.pop().unwrap_or_else(|| {
                n_regs += 1;
                (n_regs - 1) as u16
            });
            reg[i] = d;
            code.push(Instr { code: code_, d, a: reg[a as usize], b: reg[b as usize], c: reg[c as usize], n: k });
            // Free operands whose last use is this node.
            let mut ch: Vec<Id> = node.children().collect();
            ch.sort_unstable();
            ch.dedup();
            for x in ch {
                let r = reg[x as usize];
                if last_use[x as usize] == i as u32 && (r as usize) >= first_temp {
                    free.push(r);
                }
            }
        }
        assert!(n_regs < u16::MAX as usize, "expression too large");
        let outputs = roots.iter().map(|&r| reg[r as usize]).collect();
        Program { n_inputs, n_regs, consts, code, outputs }
    }
}

impl<T: Real> Program<T> {
    /// Converts the constants to another precision.
    pub fn cast<U: Real>(&self) -> Program<U> {
        Program {
            n_inputs: self.n_inputs,
            n_regs: self.n_regs,
            consts: self.consts.iter().map(|&(r, v)| (r, U::from_f64(v.to_f64()))).collect(),
            code: self.code.clone(),
            outputs: self.outputs.clone(),
        }
    }
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }
    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }
    pub fn n_regs(&self) -> usize {
        self.n_regs
    }
    pub fn instructions(&self) -> &[Instr] {
        &self.code
    }
    /// Register holding output `k`.
    #[inline]
    pub fn output_reg(&self, k: usize) -> usize {
        self.outputs[k] as usize
    }

    /// A register file with the constants loaded (inputs zero).
    pub fn init_regs(&self) -> Vec<T> {
        let mut r = vec![T::ZERO; self.n_regs.max(1)];
        self.load_consts(&mut r);
        r
    }
    fn load_consts(&self, regs: &mut [T]) {
        for &(i, v) in &self.consts {
            regs[i as usize] = v;
        }
    }

    /// Runs the program on `regs` (from [`Self::init_regs`], with inputs set).
    #[inline]
    pub fn run(&self, regs: &mut [T]) {
        let regs = &mut regs[..self.n_regs];
        for ins in &self.code {
            let a = regs[ins.a as usize];
            let v = match ins.code {
                Code::Bin(f) => {
                    let b = regs[ins.b as usize];
                    match f {
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
                Code::Un(f) => f.apply(a),
                Code::Powi => math::powi(a, ins.n),
                Code::Select => math::select(a, regs[ins.b as usize], regs[ins.c as usize]),
            };
            regs[ins.d as usize] = v;
        }
    }

    /// Convenience evaluation: `out[k]` = output `k` for `inputs`.
    pub fn eval(&self, inputs: &[T], out: &mut [T]) {
        const STACK: usize = 96;
        let mut stack = [T::ZERO; STACK];
        let mut heap;
        let regs: &mut [T] = if self.n_regs <= STACK {
            &mut stack[..self.n_regs]
        } else {
            heap = vec![T::ZERO; self.n_regs];
            &mut heap
        };
        self.load_consts(regs);
        regs[..self.n_inputs].copy_from_slice(&inputs[..self.n_inputs]);
        self.run(regs);
        for (k, o) in out.iter_mut().enumerate().take(self.outputs.len()) {
            *o = regs[self.outputs[k] as usize];
        }
    }

    /// Lane register file with constants splatted.
    pub fn init_lanes<const L: usize>(&self) -> Vec<[T; L]> {
        let mut r = vec![[T::ONE; L]; self.n_regs.max(1)];
        for &(i, v) in &self.consts {
            r[i as usize] = [v; L];
        }
        r
    }

    /// Runs `L` evaluations at once on a lane register file.
    pub fn run_lanes<const L: usize>(&self, regs: &mut [[T; L]]) {
        let regs = &mut regs[..self.n_regs];
        macro_rules! lanes {
            ($a:ident, $e:expr) => {{
                let $a = regs[$a as usize];
                let mut o = [T::ZERO; L];
                for l in 0..L {
                    let $a = $a[l];
                    o[l] = $e;
                }
                o
            }};
            ($a:ident, $b:ident, $e:expr) => {{
                let $a = regs[$a as usize];
                let $b = regs[$b as usize];
                let mut o = [T::ZERO; L];
                for l in 0..L {
                    let $a = $a[l];
                    let $b = $b[l];
                    o[l] = $e;
                }
                o
            }};
        }
        for ins in &self.code {
            let (a, b, c) = (ins.a, ins.b, ins.c);
            let out = match ins.code {
                Code::Bin(F2::Add) => lanes!(a, b, a + b),
                Code::Bin(F2::Sub) => lanes!(a, b, a - b),
                Code::Bin(F2::Mul) => lanes!(a, b, a * b),
                Code::Bin(F2::Div) => lanes!(a, b, a / b),
                Code::Bin(F2::Min) => lanes!(a, b, math::min(a, b)),
                Code::Bin(F2::Max) => lanes!(a, b, math::max(a, b)),
                Code::Bin(f) => lanes!(a, b, f.apply(a, b)),
                Code::Un(F1::Neg) => lanes!(a, -a),
                Code::Un(F1::Sqrt) => lanes!(a, a.sqrt()),
                Code::Un(F1::Abs) => lanes!(a, a.abs()),
                Code::Un(f) => lanes!(a, f.apply(a)),
                Code::Powi => match ins.n {
                    2 => lanes!(a, a * a),
                    3 => lanes!(a, a * a * a),
                    -1 => lanes!(a, T::ONE / a),
                    -2 => lanes!(a, T::ONE / (a * a)),
                    n => lanes!(a, math::powi(a, n)),
                },
                Code::Select => {
                    let (cv, av, bv) = (regs[a as usize], regs[b as usize], regs[c as usize]);
                    let mut o = [T::ZERO; L];
                    for l in 0..L {
                        o[l] = math::select(cv[l], av[l], bv[l]);
                    }
                    o
                }
            };
            regs[ins.d as usize] = out;
        }
    }
}
