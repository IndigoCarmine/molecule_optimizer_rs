//! User-defined energy terms written as expression strings.
//!
//! ```no_run
//! use molecule_optimizer_rs::ff::custom::{CustomNonbonded, AtomParams, InteractionGroups, WithCustomTerms};
//! use molecule_optimizer_rs::ff::mmff::MmffBuilder;
//! let lj = CustomNonbonded::new("lj", "lambda*4*eps*((sig/r)^12 - (sig/r)^6)")
//!     .per_atom(&["sig", "eps"], AtomParams::Uniform(vec![3.4, 0.1]))
//!     .combining("sig = 0.5*(sig1 + sig2); eps = sqrt(eps1*eps2)")
//!     .groups(InteractionGroups::InterFragment)
//!     .cutoff(10.0, Some(8.0));
//! let builder = WithCustomTerms::new(MmffBuilder::default()).with_global("lambda", 1.0).with(lj).unwrap();
//! ```
//!
//! # Expression language
//!
//! * **Numbers**: `2`, `1.5`, `.5`, `1e-3`, `2.5E+4`.
//! * **Names**: `[A-Za-z_][A-Za-z0-9_]*` — the geometric variable of the
//!   term (`r` for bonds and nonbonded pairs in Å, `theta` for angles and
//!   `phi` for torsions in radians, `x`, `y`, `z` for external fields in Å),
//!   per-term parameters, per-atom parameters `p1`/`p2` (nonbonded),
//!   global parameters, definitions, and the constant `pi`.
//! * **Operators** (by increasing precedence): `+ -` (left associative),
//!   `* /` (left associative), unary `-` / `+`, `^` (power, right
//!   associative; `**` is a synonym). `-x^2` is `-(x^2)`, `2^-1` is
//!   allowed, `a/b*c` is `(a/b)*c`. Integer constant exponents are
//!   evaluated by repeated multiplication, other exponents with `powf`.
//! * **Functions**: `sqrt exp log sin cos tan acos asin atan abs sign
//!   step erf erfc cosh sinh tanh` (one argument), `atan2(y, x) min max
//!   pow` (two), `select(c, a, b)` (three). `log` is the natural
//!   logarithm, `step(x)` is 1 for `x >= 0` and 0 otherwise, `select(c, a, b)`
//!   is `a` if `c != 0` and `b` otherwise, `sign(0) = 0`.
//! * **Definitions**: statements are separated by `;`. The *first*
//!   statement is the energy (`expr` or `E = expr`, where `E` is only a
//!   label); every following statement is a definition `name = expr`.
//!   Definitions may refer to each other in any order (they are resolved by
//!   dependency, not top-down); cycles, duplicate definitions and
//!   definitions that shadow a variable, parameter, global, function or
//!   `pi` are errors. Unused definitions are allowed. Example:
//!   `"E = k*dx^2; dx = r - r0"`.
//! * `#` starts a comment that runs to the end of the line.
//!
//! Errors ([`ExprError`]) carry the byte position and print the offending
//! line with a caret.
//!
//! Derivatives of `step`, `sign` and `select` conditions are taken as zero;
//! `abs`, `min`, `max` and `select` have one-sided derivatives at the
//! switching point. The simplifier assumes finite values (`0*x = 0`,
//! `x - x = 0`, `x^a * x^b = x^(a+b)`).
//!
//! # Pipeline
//!
//! source → [`parse`] (AST with positions) → [`ir::Graph`] (hash-consed
//! DAG: common subexpressions are shared, constants folded, algebraic
//! simplification in the constructors) → symbolic derivatives with respect
//! to the geometric variables and the globals (built in the same DAG, so
//! the energy and derivative programs share work) → *hoisting* of the
//! maximal subexpressions that depend only on per-term parameters (computed
//! once per term at build time) → [`program::Program`] register bytecode
//! (f64 or f32, scalar or `L`-lane batched evaluation, no allocation per
//! evaluation) and [`wgsl`] source generation.
//!
//! # Terms
//!
//! [`CustomNonbonded`], [`CustomBond`], [`CustomAngle`], [`CustomTorsion`]
//! and [`CustomExternal`] are added to any builder with
//! [`WithCustomTerms`]. Global parameters are declared on the builder and
//! can be changed after building with [`WithCustomTerms::set_global`] /
//! [`CustomTermsExt::set_global`]; [`CustomTermsExt::energy_param_derivative`]
//! gives `dE/d(global)`. Contrib labels are `"Custom:<name>"`.
//!
//! Nonbonded terms evaluate a pair list built once per force field (no
//! neighbor list; pairs beyond the cutoff are skipped at evaluation time)
//! and ignore periodic boundary conditions.

pub mod builder;
pub mod ir;
pub mod kernel;
pub mod math;
pub mod parse;
pub mod program;
pub mod terms;
pub mod wgsl;

#[cfg(test)]
mod tests;

pub use builder::{CompositeForceField, CustomTermsExt, WithCustomTerms, energy_param_derivative};
pub use kernel::{Globals, Kernel};
pub use parse::ExprError;
pub use terms::{
    AtomParamFn, AtomParams, BondedKind, CustomAngle, CustomBond, CustomBondedContrib, CustomContrib, CustomExternal,
    CustomNonbonded, CustomNonbondedContrib, CustomTerm, CustomTorsion, Cutoff, InteractionGroups, PreparedTerm,
    as_custom,
};

use std::collections::HashSet;
use std::sync::Mutex;

/// Interns `"Custom:<name>"` as a `&'static str` (labels must be static;
/// each distinct name is leaked once).
pub(crate) fn intern_label(name: &str) -> &'static str {
    static LABELS: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let label = format!("Custom:{name}");
    let mut guard = LABELS.lock().unwrap_or_else(|e| e.into_inner());
    let set = guard.get_or_insert_with(HashSet::new);
    if let Some(&s) = set.get(label.as_str()) {
        return s;
    }
    let s: &'static str = Box::leak(label.into_boxed_str());
    set.insert(s);
    s
}
