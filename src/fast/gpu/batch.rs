//! [`GpuBatch`]: many independent small systems evaluated in one dispatch.

use super::GpuContext;
use super::engine::{Engine, SysRef};
use crate::core::Molecule;
use crate::error::{Error, Result};
use crate::fast::approx::pairlist::{self, PairList};
use crate::fast::approx::{ApproxNonbonded, build_approx_nonbonded};
use crate::fast::backend::NonbondedEnergy;
use crate::fast::{ApproxOptions, Electrostatics};
use std::sync::{Arc, Mutex};

struct State {
    engine: Engine,
    lists: Vec<Arc<PairList>>,
    builds: usize,
}

/// Batch evaluation of the approximate nonbonded energy and gradient of many
/// independent systems (e.g. conformers of one or several molecules) in a
/// single GPU submission.
///
/// Systems are described by templates (molecule + count): all systems of a
/// template share the parameters, exclusions and type tables; each system
/// has its own coordinates and cluster-pair list (rebuilt lazily when one of
/// its atoms moved more than `buffer / 2`). Electrostatics: `Full` or
/// `ReactionField` (PME is single-system only, see
/// [`super::WgpuNonbonded`]). All templates use the same inner builder and
/// options.
pub struct GpuBatch {
    templates: Vec<ApproxNonbonded>,
    sys_tmpl: Vec<usize>,
    half_buffer2: f64,
    state: Mutex<State>,
}

impl std::fmt::Debug for GpuBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBatch").field("templates", &self.templates.len()).field("systems", &self.sys_tmpl.len()).finish()
    }
}

impl GpuBatch {
    /// `systems`: `(molecule, number of systems)` pairs; the systems are
    /// numbered in this order. The molecules' own coordinates only seed the
    /// initial pair lists.
    pub fn new(ctx: &GpuContext, systems: &[(&Molecule, usize)], inner_builder: &dyn std::any::Any, opts: &ApproxOptions) -> Result<GpuBatch> {
        if matches!(opts.electrostatics, Electrostatics::Pme { .. }) {
            return Err(Error::InvalidArgument("GpuBatch does not support PME (use WgpuNonbonded per system)".into()));
        }
        let mut templates = Vec::new();
        let mut sys_tmpl = Vec::new();
        let mut initial = Vec::new();
        for (t, &(mol, count)) in systems.iter().enumerate() {
            let nb = build_approx_nonbonded(mol, inner_builder, opts, true)?;
            initial.push(nb.current_list(&mol.flat_coords()));
            templates.push(nb);
            sys_tmpl.extend(std::iter::repeat_n(t, count));
        }
        if sys_tmpl.is_empty() {
            return Err(Error::InvalidArgument("GpuBatch needs at least one system".into()));
        }
        let refs: Vec<&ApproxNonbonded> = templates.iter().collect();
        let engine = Engine::new(ctx, &refs, false)?;
        let lists = sys_tmpl.iter().map(|&t| initial[t].clone()).collect();
        Ok(GpuBatch { half_buffer2: 0.25 * opts.buffer * opts.buffer, templates, sys_tmpl, state: Mutex::new(State { engine, lists, builds: 0 }) })
    }

    /// Number of systems.
    pub fn num_systems(&self) -> usize {
        self.sys_tmpl.len()
    }
    /// Number of atoms of system `s`.
    pub fn num_atoms(&self, s: usize) -> usize {
        self.templates[self.sys_tmpl[s]].num_atoms()
    }
    /// CPU backend of the template of system `s` (for reference evaluations).
    pub fn template(&self, s: usize) -> &ApproxNonbonded {
        &self.templates[self.sys_tmpl[s]]
    }
    /// Number of per-system pair-list rebuilds so far.
    pub fn num_rebuilds(&self) -> usize {
        self.lock().builds
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Energies of all systems (`pos[s]` = coordinates of system `s`) into
    /// `energies`; if `grads` is given, the gradient of system `s` is *added*
    /// to `grads[s]`.
    pub fn evaluate(&self, pos: &[&[f64]], energies: &mut [NonbondedEnergy], grads: Option<&mut [&mut [f64]]>) {
        let ns = self.num_systems();
        assert!(pos.len() == ns && energies.len() == ns, "one position vector and energy slot per system");
        let mut st = self.lock();
        let st = &mut *st;
        for s in 0..ns {
            let t = &self.templates[self.sys_tmpl[s]];
            assert!(pos[s].len() >= 3 * t.num_atoms(), "position vector of system {s} too short");
            let l = &st.lists[s];
            if !l.all_pairs && l.max_displacement2(pos[s]) > self.half_buffer2 {
                st.lists[s] = Arc::new(pairlist::build(&t.p, pos[s], t.cell.as_ref(), t.rlist, t.all_pairs));
                st.builds += 1;
            }
        }
        let sys: Vec<SysRef> = (0..ns)
            .map(|s| {
                let t = self.sys_tmpl[s];
                SysRef { nb: &self.templates[t], tmpl: t, list: st.lists[s].clone(), pos: pos[s] }
            })
            .collect();
        let want_grad = grads.is_some();
        st.engine.evaluate(&sys, want_grad);
        energies.copy_from_slice(&st.engine.energies);
        if let Some(gs) = grads {
            assert_eq!(gs.len(), ns, "one gradient vector per system");
            for (s, g) in gs.iter_mut().enumerate() {
                let n = self.num_atoms(s);
                let off = 3 * st.engine.atom_offset(s);
                for (o, &v) in g[..3 * n].iter_mut().zip(&st.engine.r_grad[off..off + 3 * n]) {
                    *o += v as f64;
                }
            }
        }
    }

    /// Convenience wrapper: `(energy, gradient)` of every system.
    pub fn energy_grad(&self, pos: &[&[f64]]) -> Vec<(NonbondedEnergy, Vec<f64>)> {
        let ns = self.num_systems();
        let mut e = vec![NonbondedEnergy::default(); ns];
        let mut g: Vec<Vec<f64>> = (0..ns).map(|s| vec![0.0; 3 * self.num_atoms(s)]).collect();
        {
            let mut refs: Vec<&mut [f64]> = g.iter_mut().map(|v| v.as_mut_slice()).collect();
            self.evaluate(pos, &mut e, Some(&mut refs));
        }
        e.into_iter().zip(g).collect()
    }

    /// Energies only.
    pub fn energies(&self, pos: &[&[f64]]) -> Vec<NonbondedEnergy> {
        let mut e = vec![NonbondedEnergy::default(); self.num_systems()];
        self.evaluate(pos, &mut e, None);
        e
    }
}
