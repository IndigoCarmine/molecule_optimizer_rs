//! WGSL code generation from the IR.
//!
//! Every live operation node becomes one `let` binding (the DAG sharing is
//! preserved), leaves are inlined as names or literals. The generated
//! function computes all requested roots in `f32` and returns them as a
//! scalar or `vecN<f32>`.

use super::ir::{F1, F2, Graph, Id, Node};
use std::fmt::Write;

/// Helper functions referenced by generated code (`mo_powi`, `mo_erf`,
/// `mo_erfc`). [`super::Kernel::to_wgsl`] prepends them; when several
/// generated functions go into one module, include this text once and use
/// [`super::Kernel::to_wgsl_function`].
pub const HELPERS: &str = "fn mo_powi(x: f32, n: i32) -> f32 {
    var b = x;
    var e = abs(n);
    var r = 1.0;
    loop {
        if (e == 0) { break; }
        if ((e & 1) == 1) { r = r * b; }
        b = b * b;
        e = e >> 1u;
    }
    if (n < 0) { return 1.0 / r; }
    return r;
}

fn mo_erfc(x: f32) -> f32 {
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let r = p * exp(-z * z);
    return select(2.0 - r, r, x >= 0.0);
}

fn mo_erf(x: f32) -> f32 {
    return 1.0 - mo_erfc(x);
}
";

/// WGSL `f32` literal.
pub fn literal(v: f64) -> String {
    let f = v as f32;
    if f.is_nan() {
        return "(0.0 / 0.0)".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "3.4028235e38".to_string() } else { "(-3.4028235e38)".to_string() };
    }
    let s = format!("{:?}", f.abs());
    let s = if s.contains('.') || s.contains('e') { s } else { format!("{s}.0") };
    if f < 0.0 { format!("(-{s})") } else { s }
}

/// Emits `fn name(inputs...: f32) -> ret { ... }` computing `roots` of `g`.
/// `inputs[k]` is the WGSL name of variable `k`.
pub fn emit_function(g: &Graph, roots: &[Id], name: &str, inputs: &[String]) -> Result<String, String> {
    if roots.is_empty() || roots.len() > 4 {
        return Err(format!("WGSL functions return 1 to 4 values ({} requested)", roots.len()));
    }
    let live = g.reachable(roots);
    let mut out = String::new();
    let params: Vec<String> = inputs.iter().map(|n| format!("{n}: f32")).collect();
    let ret = if roots.len() == 1 { "f32".to_string() } else { format!("vec{}<f32>", roots.len()) };
    let _ = writeln!(out, "fn {name}({}) -> {ret} {{", params.join(", "));
    let operand = |id: Id| -> String {
        match g.node(id) {
            Node::Const(b) => literal(f64::from_bits(b)),
            Node::Var(k) => inputs[k as usize].clone(),
            _ => format!("t{id}"),
        }
    };
    for i in 0..g.len() {
        if !live[i] {
            continue;
        }
        let e = match g.node(i as Id) {
            Node::Const(_) | Node::Var(_) => continue,
            Node::Un(f, a) => {
                let a = operand(a);
                match f {
                    F1::Neg => format!("-{a}"),
                    F1::Step => format!("select(0.0, 1.0, {a} >= 0.0)"),
                    F1::Erf => format!("mo_erf({a})"),
                    F1::Erfc => format!("mo_erfc({a})"),
                    f => format!("{}({a})", f.name()),
                }
            }
            Node::Bin(f, a, b) => {
                let (a, b) = (operand(a), operand(b));
                match f {
                    F2::Add | F2::Sub | F2::Mul | F2::Div => format!("{a} {} {b}", f.name()),
                    F2::Pow => format!("pow({a}, {b})"),
                    F2::Atan2 => format!("atan2({a}, {b})"),
                    F2::Min => format!("min({a}, {b})"),
                    F2::Max => format!("max({a}, {b})"),
                }
            }
            Node::Powi(a, n) => {
                let a = operand(a);
                match n {
                    2 => format!("{a} * {a}"),
                    3 => format!("{a} * {a} * {a}"),
                    -1 => format!("1.0 / {a}"),
                    -2 => format!("1.0 / ({a} * {a})"),
                    n => format!("mo_powi({a}, {n})"),
                }
            }
            Node::Select(c, a, b) => format!("select({}, {}, {} != 0.0)", operand(b), operand(a), operand(c)),
        };
        let _ = writeln!(out, "    let t{i}: f32 = {e};");
    }
    let vals: Vec<String> = roots.iter().map(|&r| operand(r)).collect();
    if roots.len() == 1 {
        let _ = writeln!(out, "    return {};", vals[0]);
    } else {
        let _ = writeln!(out, "    return {ret}({});", vals.join(", "));
    }
    out.push_str("}\n");
    Ok(out)
}

impl super::Kernel {
    /// WGSL argument names: geometric variables keep their names, per-term
    /// parameters and globals are prefixed with `p_` (avoids clashes with
    /// WGSL keywords and built-ins).
    pub fn wgsl_input_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.geometry_names().to_vec();
        v.extend(self.param_names().iter().map(|p| format!("p_{p}")));
        v.extend(self.global_names().iter().map(|p| format!("p_{p}")));
        v
    }
    /// `fn name(geom..., p_params..., p_globals...) -> vecN<f32>` returning
    /// `(E, dE/dgeom...)` (`vec2` for r/theta/phi, `vec4` for x,y,z),
    /// without the helper functions (see [`HELPERS`]).
    pub fn to_wgsl_function(&self, name: &str) -> Result<String, String> {
        let (g, roots) = self.graph();
        // Integer powers become shared multiplication chains.
        let (g, roots) = g.expand_powers(&roots[..1 + self.n_geom()]);
        emit_function(&g, &roots, name, &self.wgsl_input_names())
    }
    /// [`HELPERS`] followed by [`Self::to_wgsl_function`]: a self-contained
    /// WGSL module.
    pub fn to_wgsl(&self, name: &str) -> Result<String, String> {
        Ok(format!("{HELPERS}\n{}", self.to_wgsl_function(name)?))
    }
}
