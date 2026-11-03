//! 2D depiction coordinates: a port of RDKit's `RDDepict::compute2DCoords`
//! (RDKit 2026.09.1, `Code/GraphMol/Depictor`), the RDKit-native depictor
//! that `rdDepictor.Compute2DCoords` runs when CoordGen is not preferred.
//!
//! ```text
//! 1) find rings and fused ring systems, embed each system as regular polygons
//! 2) embed stereo (E/Z) double bonds
//! 3) grow the largest piece atom by atom / by merging the other pieces
//! 4) remove collisions (bond and spiro flips, opening angles, shortening
//!    bonds, path angle expansion), canonicalize the orientation and lay the
//!    disconnected fragments out side by side
//! ```
//!
//! The atom ordering decisions use the atom ranks RDKit uses (`_CIPRank`
//! from the legacy stereo perception, or atomic number / degree), so the
//! coordinates agree with RDKit's for the same molecule.
//!
//! Not ported: ring-system templates (`useRingTemplates`), non-tetrahedral
//! (square planar, trigonal bipyramidal, octahedral) stereo layouts and the
//! distance-matrix-mimicking variant (`compute2DCoordsMimicDistMat`).
//!
//! Large parts of this module are ports of the RDKit, BSD-3-Clause,
//! Copyright (C) 2003-2025 Greg Landrum and other RDKit contributors.

mod efrag;
mod geom;
mod utils;

use crate::core::{BondOrder, BondStereo, Molecule};
use crate::error::{Error, Result};
use crate::perception::{has_3d_coords, legacy_stereo, rings::RingInfo};
use efrag::{EmbeddedFrag, FragList};
use geom::Point2D;
use std::collections::BTreeMap;
use utils::{Ctx, MinStd};

/// RDKit's default depiction bond length (`RDDepict::BOND_LEN`).
pub const DEFAULT_BOND_LENGTH: f64 = 1.5;

/// Options of [`compute_2d_coords`]; the defaults are those of RDKit's
/// Python `rdDepictor.Compute2DCoords`.
#[derive(Debug, Clone, PartialEq)]
pub struct Compute2DCoordsParams {
    /// Atoms with fixed 2D positions (`coordMap`).
    pub coord_map: BTreeMap<usize, [f64; 2]>,
    /// Rotate each fragment so that its principal axis lies along x
    /// (`canonOrient`; ignored when `coord_map` is not empty).
    pub canon_orient: bool,
    /// Rotatable bonds flipped per random sample (`nFlipsPerSample`).
    pub n_flips_per_sample: u32,
    /// Number of random samples (`nSample`); with 0 the collisions are removed
    /// by flipping bonds on the paths between colliding atoms instead.
    pub n_samples: u32,
    /// Seed of the random sampling (`sampleSeed`; values <= 0 use RDKit's
    /// initial generator state).
    pub sample_seed: i32,
    /// Also permute the bonds of degree-4 chain atoms while sampling.
    pub permute_deg4_nodes: bool,
    /// Depiction bond length (`bondLength`).
    pub bond_length: f64,
    /// Final path angle expansion pass (`usePathAngleExpansion`).
    pub use_path_angle_expansion: bool,
}

impl Default for Compute2DCoordsParams {
    fn default() -> Self {
        Compute2DCoordsParams {
            coord_map: BTreeMap::new(),
            canon_orient: true,
            n_flips_per_sample: 0,
            n_samples: 0,
            sample_seed: 0,
            permute_deg4_nodes: false,
            bond_length: DEFAULT_BOND_LENGTH,
            use_path_angle_expansion: true,
        }
    }
}

/// Computes 2D depiction coordinates (z = 0), one per atom.
///
/// The molecule should be sanitized (hybridization and ring information are
/// used). Its stereo is taken as RDKit would see it after reading the
/// molecule: for 3D coordinates the legacy stereo perception from the
/// coordinates, otherwise the molecule's chiral tags and double-bond stereo.
pub fn compute_2d_coords(mol: &Molecule, params: &Compute2DCoordsParams) -> Result<Vec<[f64; 3]>> {
    let n = mol.num_atoms();
    if let Some((&bad, _)) = params.coord_map.iter().find(|&(&a, _)| a >= n) {
        return Err(Error::InvalidArgument(format!("coord_map atom index {bad} out of range ({n} atoms)")));
    }
    if !params.bond_length.is_finite() || params.bond_length <= 0.0 {
        return Err(Error::InvalidArgument(format!("bond_length must be positive, got {}", params.bond_length)));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // the molecule as RDKit's depictor sees it: stereo + `_CIPRank`
    let mut cp = mol.clone();
    if cp.ring_info().is_none() {
        cp.rings();
    }
    let cip = if has_3d_coords(&cp) {
        legacy_stereo::assign_stereo_legacy_from_3d_with_ranks(&mut cp)
    } else {
        legacy_stereo::legacy_cip_ranks_for_assigned_stereo(&cp, cp.ring_info().unwrap())
    };
    let ctx = Ctx { mol: &cp, ri: RingInfo::compute_including_dative(&cp), cip, bond_len: params.bond_length };

    let coord_map: BTreeMap<usize, Point2D> =
        params.coord_map.iter().map(|(&a, p)| (a, Point2D::new(p[0], p[1]))).collect();
    let mut efrags = compute_initial_coords(&ctx, &coord_map);

    let dmat = utils::distance_matrix(&cp);
    let mut rng = MinStd::new(42);
    for f in efrags.iter_mut() {
        if params.n_samples > 0 && params.n_flips_per_sample > 0 {
            f.random_sample_flips_and_permutations(
                &ctx,
                params.n_flips_per_sample,
                params.n_samples,
                params.sample_seed,
                params.permute_deg4_nodes,
                &mut rng,
            );
        } else {
            f.remove_collisions_bond_and_spiro_flip(&ctx, &dmat);
        }
    }
    for f in efrags.iter_mut() {
        f.remove_collisions_open_angles(&ctx, &dmat);
        f.remove_collisions_shorten_bonds(&ctx, &dmat);
        if params.use_path_angle_expansion {
            f.remove_collisions_path_angle_expansion(&ctx, &dmat);
        }
    }
    if coord_map.is_empty() && params.canon_orient {
        for f in efrags.iter_mut() {
            f.canonicalize_orientation();
        }
    }
    shift_coords(&mut efrags);

    let mut out = vec![[0.0; 3]; n];
    for f in &efrags {
        for (&aid, e) in &f.eatoms {
            out[aid] = [e.loc.x, e.loc.y, 0.0];
        }
    }
    // a single-atom coordMap only translates the result
    if coord_map.len() == 1 {
        let (&a, p) = coord_map.iter().next().unwrap();
        let dx = p.x - out[a][0];
        let dy = p.y - out[a][1];
        for c in out.iter_mut() {
            c[0] += dx;
            c[1] += dy;
        }
    }
    Ok(out)
}

/// [`compute_2d_coords`] and store the result in `mol.coords`.
pub fn set_2d_coords(mol: &mut Molecule, params: &Compute2DCoordsParams) -> Result<()> {
    mol.coords = compute_2d_coords(mol, params)?;
    Ok(())
}

/// `computeInitialCoords`: one finished fragment per connected component.
fn compute_initial_coords(ctx: &Ctx, coord_map: &BTreeMap<usize, Point2D>) -> Vec<EmbeddedFrag> {
    let mol = ctx.mol;
    let n = mol.num_atoms();
    let atom_ranks: Vec<i64> = (0..n).map(|a| utils::atom_depict_rank(mol, a)).collect();
    let mut efrags: FragList = Vec::new();

    let pre_spec = coord_map.len() > 1;
    if pre_spec {
        efrags.push(Some(EmbeddedFrag::from_coord_map(ctx, coord_map)));
    }

    // fused ring systems
    let arings = &ctx.ri.atom_rings;
    if !arings.is_empty() {
        let map = utils::ring_neighbor_map(arings);
        let mut done = vec![false; arings.len()];
        let mut curr = 0;
        loop {
            let fused = utils::pick_fused_rings(curr, &map, &mut done);
            let frings: Vec<Vec<usize>> = fused.iter().map(|&r| arings[r].clone()).collect();
            let mut f = EmbeddedFrag::fused_rings(ctx, &frings);
            f.setup_new_neighs(ctx);
            efrags.push(Some(f));
            match done.iter().position(|&d| !d) {
                Some(r) => curr = r,
                None => break,
            }
        }
    }

    // stereo double bonds outside rings
    for (b, bond) in mol.bonds().iter().enumerate() {
        if bond.order == BondOrder::Double
            && bond.stereo != BondStereo::None
            && ctx.num_bond_rings(b) == 0
            && bond.stereo_atoms.is_some()
        {
            let mut f = EmbeddedFrag::cis_trans(ctx, b);
            f.setup_new_neighs(ctx);
            efrags.push(Some(f));
        }
    }

    let mut embedded = vec![false; n];
    for f in efrags.iter().flatten() {
        for &a in f.eatoms.keys() {
            embedded[a] = true;
        }
    }
    let mut nratms: Vec<usize> = (0..n).filter(|&a| !embedded[a]).collect();

    let largest = |efrags: &FragList| -> Option<usize> {
        let mut best = None;
        let mut msiz = 0;
        for (i, f) in efrags.iter().enumerate() {
            if let Some(f) = f
                && !f.done
                && f.size() > msiz
            {
                msiz = f.size();
                best = Some(i);
            }
        }
        best
    };
    let mut mri = if pre_spec { Some(0) } else { largest(&efrags) };
    while mri.is_some() || !nratms.is_empty() {
        let idx = match mri {
            Some(i) => i,
            None => {
                // start a new fragment from the lowest-ranked free atom
                let mut mrank = i64::MAX;
                let mut pos = 0;
                for (p, &a) in nratms.iter().enumerate() {
                    let rank = atom_ranks[a] * n as i64 + a as i64;
                    if rank < mrank {
                        mrank = rank;
                        pos = p;
                    }
                }
                let a = nratms.remove(pos);
                efrags.push(Some(EmbeddedFrag::single(ctx, a)));
                efrags.len() - 1
            }
        };
        let mut f = efrags[idx].take().unwrap();
        f.done = true;
        f.expand(ctx, &mut nratms, &mut efrags);
        efrags[idx] = Some(f);
        mri = largest(&efrags);
    }
    efrags.into_iter().flatten().collect()
}

/// `_shiftCoords`: lay the fragments out next to each other.
fn shift_coords(efrags: &mut [EmbeddedFrag]) {
    if efrags.is_empty() {
        return;
    }
    for f in efrags.iter_mut() {
        f.compute_box();
    }
    let (mut xmax, xmin, mut ymax, ymin) = efrags[0].bbox();
    for f in efrags[1..].iter_mut() {
        let xshift = xmax + xmin <= ymax + ymin;
        let (xp, xn, yp, yn) = f.bbox();
        let shift = if xshift {
            let s = Point2D::new(xmax + xn + 1.0, 0.0);
            xmax += xp + xn + 1.0;
            s
        } else {
            let s = Point2D::new(0.0, ymax + yn + 1.0);
            ymax += yp + yn + 1.0;
            s
        };
        f.translate(shift);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_align_maps_points() {
        let t = geom::Transform2D::align(
            Point2D::new(1.0, 1.0),
            Point2D::new(1.0, 3.0),
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
        );
        let p = t.apply(Point2D::new(2.0, 0.0));
        assert!((p.x - 1.0).abs() < 1e-12 && (p.y - 3.0).abs() < 1e-12);
    }

    #[test]
    fn minstd_matches_boost() {
        // boost::minstd_rand seeded with 1: 48271, 182605794, ...
        let mut r = MinStd::new(1);
        assert_eq!(r.next_u32(), 48271);
        assert_eq!(r.next_u32(), 182605794);
    }
}
