//! Tests of the custom-term machinery.

use super::ir::Graph;
use super::kernel::lower;
use super::parse::{parse, parse_expr, same_structure};
use super::program::Program;
use super::*;
use crate::core::{Atom, BondOrder, Molecule};
use crate::error::Result;
use crate::ff::constraints::{AngleConstraintContrib, DistanceConstraintContrib, TorsionConstraintContrib};
use crate::ff::{Constraint, ContribForceField, ForceField, ForceFieldBuilder, gradient_error};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Builder producing an empty force field (custom terms only).
#[derive(Debug)]
struct Null;
impl ForceFieldBuilder for Null {
    fn name(&self) -> &str {
        "Null"
    }
    fn build(&self, mol: &Molecule, constraints: &[Constraint]) -> Result<ContribForceField> {
        let mut ff = ContribForceField::new(mol.num_atoms(), 3);
        ff.add_constraints(&mol.coords, constraints);
        Ok(ff)
    }
}

fn molecule(coords: &[[f64; 3]], bonds: &[(usize, usize)]) -> Molecule {
    let mut m = Molecule::new();
    for &c in coords {
        m.add_atom(Atom::new(6), c);
    }
    for &(a, b) in bonds {
        m.add_bond(a, b, BondOrder::Single);
    }
    m
}

/// Two 4-atom chains (fragments) with irregular geometry.
fn two_chains() -> Molecule {
    let c = [
        [0.0, 0.0, 0.0],
        [1.52, 0.1, -0.05],
        [2.05, 1.45, 0.2],
        [3.55, 1.5, 0.45],
        [0.3, 3.9, 1.1],
        [1.7, 4.3, 0.8],
        [2.4, 3.2, 2.0],
        [3.9, 3.5, 2.2],
    ];
    molecule(&c, &[(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)])
}

fn num_dfdx(f: &dyn Fn(f64) -> f64, x: f64) -> f64 {
    // Richardson-extrapolated central difference.
    let d = |h: f64| (f(x + h) - f(x - h)) / (2.0 * h);
    let h = 1e-3;
    (4.0 * d(h / 2.0) - d(h)) / 3.0
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

// ---------------------------------------------------------------------------
// parser / printer
// ---------------------------------------------------------------------------

#[test]
fn parser_round_trips() {
    let cases = [
        "4*eps*((sig/r)^12 - (sig/r)^6)",
        "-x^2",
        "(-x)^2",
        "a - (b - c)",
        "a - b - c",
        "a/(b*c)",
        "a/b*c",
        "2^3^2",
        "(2^3)^2",
        "x^-2",
        "-(-x)",
        "a - -b",
        "select(step(r - 3), 0, 0.5*k*(r - 3)^2)",
        "atan2(y, x) + min(a, b)*max(a, 1e-7)",
        "1.5e10 + .25 + 3E-4",
        "sqrt(exp(log(sin(cos(tan(acos(asin(atan(abs(erf(erfc(cosh(sinh(tanh(x)))))))))))))))",
    ];
    for s in cases {
        let e = parse_expr(s).unwrap();
        let p1 = e.to_string();
        let e2 = parse_expr(&p1).unwrap_or_else(|err| panic!("reparse of {p1:?} failed: {err}"));
        assert!(same_structure(&e, &e2), "{s} -> {p1}");
        assert_eq!(p1, e2.to_string(), "printer is not a fixed point for {s}");
    }
    // Precedence checks via evaluation.
    let ev = |s: &str| {
        let (g, r) = lower(&parse(s).unwrap(), &["x"]).unwrap();
        g.eval(r, &[3.0])
    };
    assert_eq!(ev("-x^2"), -9.0);
    assert_eq!(ev("2^3^2"), 512.0);
    assert_eq!(ev("x^-1"), 1.0 / 3.0);
    assert_eq!(ev("12/x*2"), 8.0);
    assert_eq!(ev("x - 1 - 1"), 1.0);
    assert_eq!(ev("x**2"), 9.0);
    assert_eq!(ev("2*pi"), 2.0 * std::f64::consts::PI);
    // Scripts.
    let s = parse("E = a*dx^2; dx = r - r0 # comment\n; a = 2").unwrap();
    assert_eq!(s.to_string(), "E = a*dx^2; dx = r - r0; a = 2");
}

#[test]
fn definitions_in_any_order() {
    let (g, r) = lower(&parse("E = a*dx^2; dx = r - r0; a = 2*b; b = 0.5").unwrap(), &["r", "r0"]).unwrap();
    assert_eq!(g.eval(r, &[3.0, 1.0]), 4.0);
    let (g, r) = lower(&parse("dx^2 + y; y = dx; dx = r").unwrap(), &["r"]).unwrap();
    assert_eq!(g.eval(r, &[3.0]), 12.0);
}

fn err_of(src: &str, vars: &[&str]) -> ExprError {
    match parse(src) {
        Err(e) => e,
        Ok(s) => lower(&s, vars).err().unwrap_or_else(|| panic!("{src} should fail")),
    }
}

#[test]
fn error_messages_have_positions() {
    let cases: &[(&str, usize, &str)] = &[
        ("4*eps*((sig/r)^12", 17, "expected ')'"),
        ("2*/r", 2, "expected a number"),
        ("r + q", 4, "unknown variable 'q'"),
        ("foo(r)", 0, "unknown function 'foo'"),
        ("sqrt(r, 2)", 0, "takes 1 argument"),
        ("1.5e+", 3, "malformed exponent"),
        ("r $ 2", 2, "unexpected character '$'"),
        ("x; x = r; x = 2", 10, "defined twice"),
        ("a; a = b; b = a", 14, "cyclic definition"),
        ("r = 2", 0, "shadows"),
        ("x; 2*r", 3, "only the first statement"),
        ("", 0, "empty expression"),
        ("r r", 2, "expected ';'"),
        ("sqrt", 0, "without arguments"),
        ("a; pi = 3", 3, "built-in"),
        ("2a", 1, "after number"),
    ];
    for &(src, pos, msg) in cases {
        let e = err_of(src, &["r"]);
        assert_eq!(e.pos, pos, "{src}: {e}");
        assert!(e.msg.contains(msg), "{src}: got '{}'", e.msg);
        let shown = e.to_string();
        assert!(shown.contains('^') && shown.contains(&format!("at position {pos}")), "{shown}");
    }
    // Caret is placed under the offending column of the right line.
    let e = err_of("E = r + y\n; y = q", &["r"]);
    let s = e.to_string();
    let lines: Vec<&str> = s.lines().collect();
    assert!(lines[1].ends_with("; y = q") && lines[2].trim_start() == "^", "{s}");
    assert_eq!(lines[2].len(), lines[1].len(), "{s}");
}

// ---------------------------------------------------------------------------
// IR / derivatives / programs
// ---------------------------------------------------------------------------

#[test]
fn cse_and_constant_folding() {
    let k = Kernel::compile("4*eps*((sig/r)^12 - (sig/r)^6)", &["r"], &["sig", "eps"], &Globals::new()).unwrap();
    let (g, roots) = k.graph();
    // sig/r appears once; (sig/r)^6 and ^12 share the base.
    assert!(g.op_count(&roots[..1]) <= 6, "{}", k.energy_source());
    let (g2, r2) = lower(&parse("2*3 + x*(1 + 1) - 0*y + x^1*x^2 - x*x^2").unwrap(), &["x", "y"]).unwrap();
    assert_eq!(g2.to_source(r2, &|k| ["x", "y"][k as usize].to_string()), "6 + 2*x");
    assert_eq!(g2.variables(r2), [0]);
    // Hoisting: everything except the r-dependent part is precomputed per term.
    assert!(k.n_hoisted() <= 3, "hoisted {}", k.n_hoisted());
    let mut h = vec![0.0; k.n_hoisted()];
    k.hoist(&[3.4, 0.2], &mut h);
    let mut regs = k.energy_grad_program().init_regs();
    regs[0] = 3.9;
    regs[1..1 + h.len()].copy_from_slice(&h);
    k.energy_grad_program().run(&mut regs);
    let r = 3.9f64;
    let e = 4.0 * 0.2 * ((3.4 / r).powi(12) - (3.4 / r).powi(6));
    assert!(close(regs[k.energy_grad_program().output_reg(0)], e, 1e-14));
}

/// Random smooth expression in `x`, `p`, `lam`.
fn random_expr(rng: &mut ChaCha8Rng, depth: u32) -> String {
    if depth == 0 || rng.gen_bool(0.2) {
        return match rng.gen_range(0..5) {
            0 | 1 => "x".into(),
            2 => "p".into(),
            3 => "lam".into(),
            _ => format!("{:.2}", rng.gen_range(0.2..2.0)),
        };
    }
    let a = random_expr(rng, depth - 1);
    match rng.gen_range(0..26) {
        0 => format!("({a}) + ({})", random_expr(rng, depth - 1)),
        1 => format!("({a}) - ({})", random_expr(rng, depth - 1)),
        2 | 3 => format!("({a})*({})", random_expr(rng, depth - 1)),
        4 => format!("({a})/(1.5 + ({})^2)", random_expr(rng, depth - 1)),
        5 => format!("({a})^2"),
        6 => format!("({a})^3"),
        7 => format!("(1.2 + ({a})^2)^(0.3*({}))", random_expr(rng, depth - 1)),
        8 => format!("sqrt(1.5 + ({a})^2)"),
        9 => format!("exp(0.3*tanh({a}))"),
        10 => format!("log(2 + ({a})^2)"),
        11 => format!("sin({a})"),
        12 => format!("cos({a})"),
        13 => format!("tan(0.5*tanh({a}))"),
        14 => format!("acos(0.9*tanh({a}))"),
        15 => format!("asin(0.9*tanh({a}))"),
        16 => format!("atan({a})"),
        17 => format!("atan2({a}, 2 + ({})^2)", random_expr(rng, depth - 1)),
        18 => format!("erf({a})"),
        19 => format!("erfc({a})"),
        20 => format!("cosh(0.5*tanh({a}))"),
        21 => format!("sinh(tanh({a}))"),
        22 => format!("tanh({a})"),
        23 => format!("-({a})"),
        24 => format!("({a})^-1.5 * 0 + ({a})^(-2) * 0.1 + 1/(1 + ({a})^2)"),
        _ => format!("pow(1.1 + ({a})^2, 1.7)"),
    }
}

#[test]
fn symbolic_derivatives_match_finite_differences() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let globals = Globals::new().with("lam", 0.7);
    let mut checked = 0;
    for _ in 0..300 {
        let src = random_expr(&mut rng, 4);
        let k = Kernel::compile(&src, &["x"], &["p"], &globals).unwrap_or_else(|e| panic!("{src}: {e}"));
        for _ in 0..3 {
            let x: f64 = rng.gen_range(-1.5..1.5);
            let p: f64 = rng.gen_range(0.3..1.5);
            let lam = 0.7;
            let lam_in: Vec<f64> = if k.n_globals() == 1 { vec![lam] } else { vec![] };
            let reference = |x: f64, p: f64, l: f64| {
                let gl: Vec<f64> = if k.n_globals() == 1 { vec![l] } else { vec![] };
                k.eval_reference(&[x], &[p], &gl)[0]
            };
            let e0 = reference(x, p, lam);
            if !e0.is_finite() || e0.abs() > 1e8 {
                continue;
            }
            // Compiled program == tree evaluation, bit for bit is not
            // required (hoisting reassociates nothing, but folding can);
            // require 1e-13.
            let mut h = vec![0.0; k.n_hoisted()];
            k.hoist(&[p], &mut h);
            let mut inp = vec![x];
            inp.extend(&h);
            inp.extend(&lam_in);
            let mut out = [0.0; 2];
            k.energy_grad_program().eval(&inp, &mut out);
            assert!(close(out[0], e0, 1e-12), "{src}: {} vs {e0}", out[0]);
            let fd = num_dfdx(&|x| reference(x, p, lam), x);
            assert!(
                close(out[1], fd, 1e-6),
                "{src} at x={x}: d/dx {} vs fd {fd}\n  d = {}",
                out[1],
                k.derivative_source(0)
            );
            if k.n_globals() == 1 {
                let mut d = [0.0];
                k.global_derivative_program().eval(&inp, &mut d);
                let fd = num_dfdx(&|l| reference(x, p, l), lam);
                assert!(close(d[0], fd, 1e-6), "{src}: d/dlam {} vs fd {fd}", d[0]);
            }
            // Printed simplified expression evaluates to the same value.
            let printed = k.energy_source();
            let (g2, r2) = lower(&parse(&printed).unwrap(), &["x", "p", "lam"]).unwrap();
            assert!(close(g2.eval(r2, &[x, p, lam]), e0, 1e-12), "{src} printed as {printed}");
            checked += 1;
        }
    }
    assert!(checked > 600, "only {checked} points checked");
}

#[test]
fn every_function_differentiates() {
    let cases = [
        ("sqrt(0.3 + 0.4*x)", 0.37),
        ("exp(0.3 + 0.4*x)", 0.37),
        ("log(0.3 + 0.4*x)", 0.37),
        ("sin(0.3 + 0.4*x)", 0.37),
        ("cos(0.3 + 0.4*x)", 0.37),
        ("tan(0.3 + 0.4*x)", 0.37),
        ("acos(0.3 + 0.4*x)", 0.37),
        ("asin(0.3 + 0.4*x)", 0.37),
        ("atan(0.3 + 0.4*x)", 0.37),
        ("abs(0.3 - 0.4*x)", 0.37),
        ("abs(0.3 - 0.4*x)", 1.37),
        ("erf(0.3 + 0.4*x)", 0.37),
        ("erfc(0.3 + 0.4*x)", 0.37),
        ("erfc(3*x)", 1.1),
        ("cosh(0.3 + 0.4*x)", 0.37),
        ("sinh(0.3 + 0.4*x)", 0.37),
        ("tanh(0.3 + 0.4*x)", 0.37),
        ("atan2(x, 0.4 + x^2)", 0.37),
        ("atan2(0.4, x)", -0.37),
        ("min(x^2, 0.5)", 0.3),
        ("min(x^2, 0.5)", 0.9),
        ("max(x^2, 0.5)", 0.3),
        ("max(x^2, 0.5)", 0.9),
        ("pow(x, 1.7)", 0.8),
        ("pow(1.3, x)", 0.8),
        ("x^x", 0.8),
        ("select(step(x - 0.5), x^2, 3*x)", 0.3),
        ("select(step(x - 0.5), x^2, 3*x)", 0.7),
        ("sign(x - 0.5)*x^2 + step(x)", 0.3),
        ("x^-3 + x^7", 0.8),
        ("1/x + x/(1 + x)", 0.8),
    ];
    for (src, x) in cases {
        let k = Kernel::compile(src, &["x"], &[], &Globals::new()).unwrap();
        let mut out = [0.0; 2];
        k.energy_grad_program().eval(&[x], &mut out);
        let fd = num_dfdx(&|x| k.eval_reference(&[x], &[], &[])[0], x);
        assert!(close(out[1], fd, 1e-8), "{src} at {x}: {} vs {fd} ({})", out[1], k.derivative_source(0));
    }
}

mod erfc_data {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/erfc_reference.rs"));
    pub(super) const REF: &[(f64, f64)] = ERFC_REF;
}

#[test]
fn erf_erfc_accuracy() {
    for &(x, r) in erfc_data::REF {
        assert!(((math::erfc(x) - r) / r).abs() < 1e-13, "erfc({x}) = {} vs {r}", math::erfc(x));
        assert!((math::erf(x) - (1.0 - r)).abs() < 1e-15, "erf({x})");
        assert!((math::erf(-x) + (1.0 - r)).abs() < 1e-15);
        assert!((math::erfc(-x) - (2.0 - r)).abs() < 1e-15);
    }
}

#[test]
fn lanes_and_f32_match_scalar() {
    let k = Kernel::compile(
        "E = lam*4*eps*(s6^2 - s6) + q/r*erfc(0.3*r) + select(step(r - 5), 0, (r-5)^2); s6 = (sig/r)^6",
        &["r"],
        &["sig", "eps", "q"],
        &Globals::new().with("lam", 0.8),
    )
    .unwrap();
    let prog = k.energy_grad_program();
    let mut h = vec![0.0; k.n_hoisted()];
    k.hoist(&[3.2, 0.15, -0.4], &mut h);
    let mut lanes = prog.init_lanes::<8>();
    let rs: Vec<f64> = (0..8).map(|i| 2.5 + 0.7 * i as f64).collect();
    for l in 0..8 {
        lanes[0][l] = rs[l];
        for (s, v) in h.iter().enumerate() {
            lanes[1 + s][l] = *v;
        }
        lanes[1 + h.len()][l] = 0.8;
    }
    prog.run_lanes(&mut lanes);
    let p32: Program<f32> = prog.cast();
    for (l, &r) in rs.iter().enumerate() {
        let mut inp = vec![r];
        inp.extend(&h);
        inp.push(0.8);
        let mut out = [0.0; 2];
        prog.eval(&inp, &mut out);
        assert_eq!(out[0], lanes[prog.output_reg(0)][l]);
        assert_eq!(out[1], lanes[prog.output_reg(1)][l]);
        let inp32: Vec<f32> = inp.iter().map(|&v| v as f32).collect();
        let mut o32 = [0.0f32; 2];
        p32.eval(&inp32, &mut o32);
        assert!(close(o32[0] as f64, out[0], 1e-4) && close(o32[1] as f64, out[1], 1e-4), "{o32:?} vs {out:?}");
    }
}

#[test]
fn register_allocation_reuses_registers() {
    // A long chain needs only a handful of temporaries.
    let mut src = String::from("x");
    for i in 0..200 {
        src = format!("sin({src} + {})", i as f64 * 0.01);
    }
    let k = Kernel::compile(&src, &["x"], &[], &Globals::new()).unwrap();
    let p = k.energy_program();
    assert!(p.n_regs() < 1 + 200 + 8, "{} registers", p.n_regs());
    let mut g = Graph::new();
    let x = g.var(0);
    let s = g.f1(ir::F1::Sin, x);
    let c = g.f1(ir::F1::Cos, x);
    let a = g.add(s, c);
    let prog = Program::compile(&g, &[a, s], 1);
    let mut out = [0.0; 2];
    prog.eval(&[0.4], &mut out);
    assert_eq!(out, [0.4f64.sin() + 0.4f64.cos(), 0.4f64.sin()]);
}

// ---------------------------------------------------------------------------
// WGSL
// ---------------------------------------------------------------------------

fn validate_wgsl(src: &str) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("WGSL parse error: {}\n{src}", e.emit_to_string(src)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation error: {e:?}\n{src}"));
}

#[test]
fn wgsl_output_parses_with_naga() {
    let globals = Globals::new().with("lambda", 1.0);
    let cases: &[(&str, &[&str], &[&str])] = &[
        ("4*eps*((sig/r)^12 - (sig/r)^6)", &["r"], &["sig", "eps"]),
        ("lambda*q1q2/r*erfc(alpha*r) + erf(r) - 1e-30*r^-7", &["r"], &["q1q2", "alpha"]),
        ("0.5*k*(r - r0)^2", &["r"], &["k", "r0"]),
        ("D*(1 - exp(-a*(r - r0)))^2", &["r"], &["D", "a", "r0"]),
        ("k*(theta - t0)^2 + select(step(theta - 2), sin(theta), cos(theta))", &["theta"], &["k", "t0"]),
        ("k*(1 + cos(n*phi - d)) + abs(phi) + atan2(phi, 2) + sign(phi)", &["phi"], &["k", "n", "d"]),
        ("k*max(0, sqrt(x^2 + y^2 + z^2) - R)^2", &["x", "y", "z"], &["k", "R"]),
        (
            "pow(r, 1.5) + min(r, 2) + tanh(r) + sinh(r) + cosh(r) + tan(r) + acos(0.5/r) + asin(0.5/r) + atan(r) + log(r)",
            &["r"],
            &[],
        ),
        ("2", &["r"], &[]),
        ("r", &["r"], &[]),
    ];
    let mut module = String::from(wgsl::HELPERS);
    for (i, (src, geom, params)) in cases.iter().enumerate() {
        let k = Kernel::compile(src, geom, params, &globals).unwrap();
        let text = k.to_wgsl("energy_and_deriv").unwrap();
        assert!(text.contains("fn energy_and_deriv("));
        if geom.len() == 1 {
            assert!(
                text.contains(&format!("fn energy_and_deriv({}: f32", geom[0])) && text.contains("-> vec2<f32>"),
                "{text}"
            );
        } else {
            assert!(text.contains("-> vec4<f32>"), "{text}");
        }
        validate_wgsl(&text);
        module.push_str(&k.to_wgsl_function(&format!("f{i}")).unwrap());
    }
    // Several functions sharing one copy of the helpers.
    validate_wgsl(&module);
    // A compute entry point calling a generated function.
    let k = Kernel::compile("4*eps*((sig/r)^12 - (sig/r)^6)", &["r"], &["sig", "eps"], &globals).unwrap();
    let full = format!(
        "{}\n@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;\n\
         @compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n    \
         let r = 3.0 + f32(id.x) * 0.01;\n    data[id.x] = lj(r, 3.4, 0.1);\n}}\n",
        k.to_wgsl("lj").unwrap()
    );
    validate_wgsl(&full);
}

// ---------------------------------------------------------------------------
// Contribs
// ---------------------------------------------------------------------------

fn build(b: &WithCustomTerms<Null>, mol: &Molecule) -> ContribForceField {
    b.build(mol, &[]).unwrap()
}

#[test]
fn custom_bond_equals_distance_constraint() {
    let mol = two_chains();
    let pos = mol.flat_coords();
    let b = WithCustomTerms::new(Null)
        .with(CustomBond::new("harm", "0.5*k*(r - r0)^2", &["k", "r0"]).term(0, 3, &[12.0, 2.5]).term(
            1,
            6,
            &[7.0, 3.0],
        ))
        .unwrap();
    let ff = build(&b, &mol);
    let mut reference = ContribForceField::new(mol.num_atoms(), 3);
    reference.add_contrib(Box::new(DistanceConstraintContrib { i: 0, j: 3, min: 2.5, max: 2.5, k: 12.0 }));
    reference.add_contrib(Box::new(DistanceConstraintContrib { i: 1, j: 6, min: 3.0, max: 3.0, k: 7.0 }));
    let (mut g1, mut g2) = (vec![0.0; pos.len()], vec![0.0; pos.len()]);
    let e1 = ff.energy_grad(&pos, &mut g1);
    let e2 = reference.energy_grad(&pos, &mut g2);
    assert!((e1 - e2).abs() < 1e-12 * e2.abs().max(1.0), "{e1} vs {e2}");
    for (a, b) in g1.iter().zip(&g2) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
    assert!(gradient_error(&ff, &pos) < 1e-6);
    assert_eq!(ff.energy_breakdown(&pos)[0].0, "Custom:harm");
}

#[test]
fn custom_angle_and_torsion_equal_constraints() {
    let mol = two_chains();
    let pos = mol.flat_coords();
    let p0 = crate::ff::utils::compute_dihedral(&pos, 0, 1, 2, 3).dihedral + 0.3;
    let b = WithCustomTerms::new(Null)
        .with(CustomAngle::new("ang", "k*((theta - t0)*180/pi)^2", &["k", "t0"]).term(0, 1, 2, &[0.3, 1.9]))
        .unwrap()
        .with(CustomTorsion::new("tor", "k*((phi - p0)*180/pi)^2", &["k", "p0"]).term(0, 1, 2, 3, &[0.2, p0]))
        .unwrap();
    let ff = build(&b, &mol);
    let mut reference = ContribForceField::new(mol.num_atoms(), 3);
    let deg = 180.0 / std::f64::consts::PI;
    reference.add_contrib(Box::new(AngleConstraintContrib {
        i: 0,
        j: 1,
        k: 2,
        min: 1.9 * deg,
        max: 1.9 * deg,
        force: 0.3,
    }));
    reference.add_contrib(Box::new(TorsionConstraintContrib {
        idx: [0, 1, 2, 3],
        min: crate::ff::utils::normalize_angle_deg(p0 * deg),
        max: crate::ff::utils::normalize_angle_deg(p0 * deg),
        force: 0.2,
    }));
    let (mut g1, mut g2) = (vec![0.0; pos.len()], vec![0.0; pos.len()]);
    let e1 = ff.energy_grad(&pos, &mut g1);
    let e2 = reference.energy_grad(&pos, &mut g2);
    assert!(e2 > 1.0);
    assert!((e1 - e2).abs() < 1e-10 * e2, "{e1} vs {e2}");
    for (a, b) in g1.iter().zip(&g2) {
        assert!((a - b).abs() < 1e-9 * e2, "{a} vs {b}");
    }
}

#[test]
fn bonded_and_external_gradients() {
    let mol = two_chains();
    let pos = mol.flat_coords();
    let b = WithCustomTerms::new(Null)
        .with_global("lambda", 0.6)
        .with(
            CustomBond::new("morse", "lambda*D*(1 - exp(-a*(r - r0)))^2", &["D", "a", "r0"])
                .term(0, 1, &[90.0, 2.0, 1.4])
                .term(5, 6, &[80.0, 1.8, 1.6]),
        )
        .unwrap()
        .with(
            CustomAngle::new("cosang", "k*(cos(theta) - cos(t0))^2 + lambda^2*theta", &["k", "t0"])
                .term(0, 1, 2, &[30.0, 1.9])
                .term(4, 5, 6, &[25.0, 2.0]),
        )
        .unwrap()
        .with(
            CustomTorsion::new("fourier", "k*(1 + cos(n*phi - d)) + lambda*sin(phi)", &["k", "n", "d"])
                .term(0, 1, 2, 3, &[1.3, 3.0, 0.2])
                .term(4, 5, 6, 7, &[0.7, 2.0, 3.1]),
        )
        .unwrap()
        .with(
            CustomExternal::new(
                "wall",
                "k*max(0, sqrt((x-1)^2 + (y-1.5)^2 + z^2) - R)^2 + lambda*0.1*x*y*z",
                &["k", "R"],
            )
            .atoms(0..8, &[5.0, 2.0]),
        )
        .unwrap()
        .with_global("cx", 1.0)
        .with_global("cy", 1.5)
        // cx/cy were declared after "wall": a term using them must come later.
        .with(CustomExternal::new("well", "lambda*k*((x - cx)^2 + (y - cy)^2)", &["k"]).atom(3, &[0.5]))
        .unwrap();
    let ff = build(&b, &mol);
    let err = gradient_error(&ff, &pos);
    assert!(err < 1e-6, "gradient error {err}");
    // dE/dlambda vs numeric.
    let d = ff.energy_param_derivative("lambda", &pos);
    let h = 1e-5;
    ff.set_global("lambda", 0.6 + h).unwrap();
    let ep = ff.energy(&pos);
    ff.set_global("lambda", 0.6 - h).unwrap();
    let em = ff.energy(&pos);
    ff.set_global("lambda", 0.6).unwrap();
    let fd = (ep - em) / (2.0 * h);
    assert!(close(d, fd, 1e-7), "dE/dlambda {d} vs {fd}");
    assert!(d.abs() > 1e-3);
    assert_eq!(ff.energy_param_derivative("unknown", &pos), 0.0);
    assert!(ff.set_global("unknown", 1.0).is_err());
    let labels: Vec<String> = ff.energy_breakdown(&pos).into_iter().map(|(k, _)| k).collect();
    assert_eq!(labels, ["Custom:morse", "Custom:cosang", "Custom:fourier", "Custom:wall", "Custom:well"]);
    // Using a global before its declaration is an error.
    let e = WithCustomTerms::new(Null).with(CustomExternal::new("w", "q*x", &[]).atom(0, &[])).unwrap_err();
    assert!(e.to_string().contains("unknown variable 'q'"), "{e}");
}

#[test]
fn set_global_updates_built_force_field() {
    let mol = two_chains();
    let pos = mol.flat_coords();
    let b = WithCustomTerms::new(Null)
        .with_global("k", 1.0)
        .with(CustomBond::new("b", "k*r^2", &[]).term(0, 1, &[]))
        .unwrap();
    let ff = b.build_dyn(&mol, &[]).unwrap();
    let e1 = ff.energy(&pos);
    b.set_global("k", 3.0).unwrap();
    assert!(close(ff.energy(&pos), 3.0 * e1, 1e-15));
    assert_eq!(b.get_global("k"), Some(3.0));
    assert!(b.set_global("nope", 1.0).is_err());
}

/// Hand-written LJ with quintic switching for reference.
fn reference_lj(pos: &[f64], pairs: &[(usize, usize, f64, f64)], rc: f64, rs: f64, grad: &mut [f64]) -> f64 {
    let mut e = 0.0;
    for &(i, j, sig, eps) in pairs {
        let d = [pos[3 * i] - pos[3 * j], pos[3 * i + 1] - pos[3 * j + 1], pos[3 * i + 2] - pos[3 * j + 2]];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if r >= rc {
            continue;
        }
        let s6 = (sig / r).powi(6);
        let v = 4.0 * eps * (s6 * s6 - s6);
        let dv = 4.0 * eps * (-12.0 * s6 * s6 + 6.0 * s6) / r;
        let (s, ds) = if r > rs {
            let t = (r - rs) / (rc - rs);
            (
                1.0 - 10.0 * t.powi(3) + 15.0 * t.powi(4) - 6.0 * t.powi(5),
                (-30.0 * t * t + 60.0 * t.powi(3) - 30.0 * t.powi(4)) / (rc - rs),
            )
        } else {
            (1.0, 0.0)
        };
        e += v * s;
        let de = dv * s + v * ds;
        for c in 0..3 {
            grad[3 * i + c] += de * d[c] / r;
            grad[3 * j + c] -= de * d[c] / r;
        }
    }
    e
}

#[test]
fn custom_lj_equals_hand_written_reference() {
    let mol = two_chains();
    let pos = mol.flat_coords();
    let sig: Vec<f64> = (0..8).map(|i| 3.0 + 0.1 * i as f64).collect();
    let eps: Vec<f64> = (0..8).map(|i| 0.05 + 0.02 * i as f64).collect();
    let table: Vec<Vec<f64>> = (0..8).map(|i| vec![sig[i], eps[i]]).collect();
    let (rc, rs) = (4.5, 3.0);
    let lj = CustomNonbonded::new("lj", "4*eps*((sig/r)^12 - (sig/r)^6)")
        .per_atom(&["sig", "eps"], AtomParams::Table(table))
        .combining("sig = 0.5*(sig1 + sig2); eps = sqrt(eps1*eps2)")
        .exclude_within_bonds(2)
        .cutoff(rc, Some(rs));
    let b = WithCustomTerms::new(Null).with(lj.clone()).unwrap();
    let ff = build(&b, &mol);
    let topo = mol.topological_distances();
    let mut pairs = Vec::new();
    for i in 0..8 {
        for j in i + 1..8 {
            if topo[i][j] > 2 {
                pairs.push((i, j, 0.5 * (sig[i] + sig[j]), (eps[i] * eps[j]).sqrt()));
            }
        }
    }
    let mut gr = vec![0.0; pos.len()];
    let er = reference_lj(&pos, &pairs, rc, rs, &mut gr);
    let mut g = vec![0.0; pos.len()];
    let e = ff.energy_grad(&pos, &mut g);
    assert!((e - er).abs() < 1e-12 * er.abs().max(1.0), "{e} vs {er}");
    assert!(g.iter().zip(&gr).all(|(a, b)| (a - b).abs() < 1e-12 * b.abs().max(1.0)), "{g:?}\n{gr:?}");
    assert!(gradient_error(&ff, &pos) < 1e-6);
    // Some pairs are inside the switching region.
    assert!(pairs.iter().any(|&(i, j, ..)| {
        let r = crate::core::geometry::dist(mol.coords[i], mol.coords[j]);
        r > rs && r < rc
    }));

    // Interaction groups and exclusions.
    let count = |nb: CustomNonbonded| {
        let p = PreparedTerm::new(nb.into(), &Globals::new()).unwrap();
        let c = p.contrib(&mol, &Globals::new()).unwrap();
        c.as_any().unwrap().downcast_ref::<CustomNonbondedContrib>().unwrap().num_pairs()
    };
    let base = || CustomNonbonded::new("c", "1/r");
    assert_eq!(count(base()), 28);
    assert_eq!(count(base().groups(InteractionGroups::InterFragment)), 16);
    assert_eq!(count(base().groups(InteractionGroups::IntraFragment)), 12);
    assert_eq!(count(base().groups(InteractionGroups::IntraFragment).exclude_within_bonds(1)), 6);
    assert_eq!(count(base().groups(InteractionGroups::IntraFragment).exclude_within_bonds(3)), 0);
    assert_eq!(count(base().exclude(0, 7).exclude(7, 0)), 27);
    assert_eq!(count(base().groups(InteractionGroups::Between(vec![0, 1], vec![1, 2, 3]))), 5);
}

#[test]
fn nonbonded_param_derivative_and_overrides() {
    let mol = two_chains();
    let pos = mol.flat_coords();
    let nb = CustomNonbonded::new("soft", "lambda^2*eps*(1 - (r/sig)^2)^2*step(sig - r) + (1 - lambda)*q1*q2/r")
        .per_atom(
            &["s", "e", "q"],
            AtomParams::Function(std::sync::Arc::new(|_m: &Molecule, i: usize| {
                vec![3.0 + 0.2 * i as f64, 0.3, if i.is_multiple_of(2) { 0.4 } else { -0.3 }]
            })),
        )
        .combining("sig = s1 + s2; eps = sqrt(e1*e2)")
        .pair_override(0, 4, &[5.5, 1.0])
        .groups(InteractionGroups::InterFragment);
    let b = WithCustomTerms::new(Null).with_global("lambda", 0.4).with(nb).unwrap();
    let ff = build(&b, &mol);
    assert!(gradient_error(&ff, &pos) < 1e-6);
    let d = ff.energy_param_derivative("lambda", &pos);
    let h = 1e-6;
    b.set_global("lambda", 0.4 + h).unwrap();
    let ep = ff.energy(&pos);
    b.set_global("lambda", 0.4 - h).unwrap();
    let em = ff.energy(&pos);
    b.set_global("lambda", 0.4).unwrap();
    assert!(close(d, (ep - em) / (2.0 * h), 1e-7), "{d} vs {}", (ep - em) / (2.0 * h));
    // Wrong lengths are reported.
    let bad = CustomNonbonded::new("x", "a1*a2/r").per_atom(&["a"], AtomParams::Uniform(vec![1.0, 2.0]));
    let e = WithCustomTerms::new(Null).with(bad).unwrap().build(&mol, &[]).unwrap_err();
    assert!(e.to_string().contains("1 expected"), "{e}");
}

#[test]
fn large_nonbonded_runs_in_parallel_consistently() {
    // > PAR_PAIRS pairs: parallel path must equal the serial reference.
    let n = 300;
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let coords: Vec<[f64; 3]> =
        (0..n).map(|_| [rng.gen_range(0.0..30.0), rng.gen_range(0.0..30.0), rng.gen_range(0.0..30.0)]).collect();
    let mol = molecule(&coords, &[]);
    let pos = mol.flat_coords();
    let b = WithCustomTerms::new(Null)
        .with(
            CustomNonbonded::new("lj", "4*eps*((sig/r)^12 - (sig/r)^6)")
                .per_atom(&["sig", "eps"], AtomParams::Uniform(vec![1.0, 0.2]))
                .combining("sig = sig1; eps = eps1"),
        )
        .unwrap();
    let ff = build(&b, &mol);
    let pairs: Vec<(usize, usize, f64, f64)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j, 1.0, 0.2))).collect();
    let mut gr = vec![0.0; pos.len()];
    let er = reference_lj(&pos, &pairs, f64::INFINITY, f64::INFINITY, &mut gr);
    let mut g = vec![0.0; pos.len()];
    let e = ff.energy_grad(&pos, &mut g);
    assert!(close(e, er, 1e-10), "{e} vs {er}");
    assert!(g.iter().zip(&gr).all(|(a, b)| close(*a, *b, 1e-9)));
}

#[test]
fn works_with_mmff_and_optimizer() {
    let mut mol = crate::io::read_molfile(include_str!("../../../examples/data/paracetamol.mol")).unwrap();
    let n0 = mol.num_atoms();
    let copy = mol.clone();
    for a in 0..n0 {
        let p = copy.coords[a];
        mol.add_atom(copy.atom(a).clone(), [p[0] + 6.0, p[1] + 1.0, p[2] + 0.5]);
    }
    for bd in copy.bonds() {
        mol.add_bond(bd.begin + n0, bd.end + n0, bd.order);
    }
    crate::perception::sanitize(&mut mol).unwrap();
    let wall = CustomExternal::new("wall", "k*max(0, sqrt(x^2 + y^2 + z^2) - R)^2", &[]).atoms(0..mol.num_atoms(), &[]);
    let lj = CustomNonbonded::new("attract", "-eps*exp(-((r - r0)/w)^2)")
        .per_atom(&["eps"], AtomParams::Uniform(vec![0.2]))
        .combining("eps = sqrt(eps1*eps2); r0 = 4; w = 1.5")
        .groups(InteractionGroups::InterFragment);
    let b = WithCustomTerms::new(crate::ff::mmff::MmffBuilder::default())
        .with_global("k", 10.0)
        .with_global("R", 6.0)
        .with(wall)
        .unwrap()
        .with(lj)
        .unwrap();
    assert_eq!(b.name(), "MMFF94+Custom");
    let ff = b.build(&mol, &[]).unwrap();
    let pos = mol.flat_coords();
    assert!(gradient_error(&ff, &pos) < 1e-4);
    let wall_e = |p: &[f64]| ff.energy_breakdown(p).into_iter().find(|(k, _)| k == "Custom:wall").unwrap().1;
    let w0 = wall_e(&pos);
    let mut opts = crate::calc::OptimizeOptions::default();
    opts.minimize.max_iters = 5000;
    let r = crate::calc::optimize::optimize(&mol, &b, &opts).unwrap();
    assert!(r.grad_rms < 1e-2, "{:?}", r.status);
    let end = crate::calc::to_flat(&r.coords);
    let bd = ff.energy_breakdown(&end);
    assert!(bd.iter().any(|(k, v)| k == "Custom:attract" && *v < 0.0), "{bd:?}");
    assert!(w0 > 1.0 && wall_e(&end) < 0.1 * w0, "wall energy {w0} -> {}", wall_e(&end));
}

/// Custom LJ over ~1e6 pairs vs a hand-written loop (prints ns/pair).
#[test]
fn performance_custom_lj_vs_hand_written() {
    let n = 1415; // 1415*1414/2 = 1,000,405 pairs
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let coords: Vec<[f64; 3]> =
        (0..n).map(|_| [rng.gen_range(0.0..40.0), rng.gen_range(0.0..40.0), rng.gen_range(0.0..40.0)]).collect();
    let mol = molecule(&coords, &[]);
    let pos = mol.flat_coords();
    let table: Vec<Vec<f64>> = (0..n).map(|i| vec![3.0 + 0.001 * i as f64, 0.1]).collect();
    let b = WithCustomTerms::new(Null)
        .with(
            CustomNonbonded::new("lj", "4*eps*((sig/r)^12 - (sig/r)^6)")
                .per_atom(&["sig", "eps"], AtomParams::Table(table.clone()))
                .combining("sig = 0.5*(sig1 + sig2); eps = sqrt(eps1*eps2)"),
        )
        .unwrap();
    let ff = build(&b, &mol);
    let c = &ff.contribs()[0];
    let np = n * (n - 1) / 2;
    let pairs: Vec<(usize, usize, f64, f64)> = (0..n)
        .flat_map(|i| {
            let t = &table;
            (i + 1..n).map(move |j| (i, j, 0.5 * (t[i][0] + t[j][0]), (t[i][1] * t[j][1]).sqrt()))
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let time = |f: &mut dyn FnMut()| {
        f();
        let reps = 3;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        t.elapsed().as_secs_f64() / reps as f64 / np as f64 * 1e9
    };
    let mut g = vec![0.0; pos.len()];
    let mut e_custom = 0.0;
    let mut e_ref = 0.0;
    let t_custom_eg = pool.install(|| {
        time(&mut || {
            g.iter_mut().for_each(|x| *x = 0.0);
            e_custom = c.energy(&pos);
            c.grad(&pos, &mut g);
        })
    });
    let t_custom_e = pool.install(|| time(&mut || e_custom = c.energy(&pos)));
    let t_custom_g = pool.install(|| {
        time(&mut || {
            g.iter_mut().for_each(|x| *x = 0.0);
            c.grad(&pos, &mut g);
        })
    });
    let k = &b.terms()[0].kernel;
    println!(
        "LJ kernel: {} hoisted values, energy {} instructions, energy+derivative {} instructions
  E = {}
  dE/dr = {}",
        k.n_hoisted(),
        k.energy_program().instructions().len(),
        k.energy_grad_program().instructions().len(),
        k.energy_source(),
        k.derivative_source(0)
    );
    let mut gr = vec![0.0; pos.len()];
    let t_ref = time(&mut || {
        gr.iter_mut().for_each(|x| *x = 0.0);
        e_ref = reference_lj(&pos, &pairs, f64::INFINITY, f64::INFINITY, &mut gr);
    });
    let t_par = time(&mut || {
        g.iter_mut().for_each(|x| *x = 0.0);
        c.grad(&pos, &mut g);
    });
    assert!(close(e_custom, e_ref, 1e-10));
    println!(
        "custom LJ, {np} pairs (1 thread): energy() {t_custom_e:.2} ns/pair, grad() (energy+gradient) {t_custom_g:.2} ns/pair, \
         energy()+grad() {t_custom_eg:.2} ns/pair; hand-written energy+gradient {t_ref:.2} ns/pair; \
         grad() with {} threads {t_par:.2} ns/pair",
        rayon::current_num_threads()
    );
}
