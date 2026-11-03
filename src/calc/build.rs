//! 3D structures from 2D drawings.

use crate::core::Molecule;
use crate::distgeom::{self, EmbedParams};
use crate::error::Result;
use crate::optim::Observer;
use crate::perception::{self, hydrogens, stereo};

/// Builds a 3D structure from a 2D drawing (such as a molfile written by a
/// structure editor, read with [`crate::io::read_molfile`]): adds the
/// hydrogens as atoms, takes the stereo the drawing shows (wedges and drawn
/// double bonds, [`stereo::assign_stereo_from_2d`]) and embeds it with
/// `params` (ETKDG, say), whose `random_seed` makes the result reproducible.
///
/// The returned molecule has every hydrogen as an atom and the embedded
/// coordinates; it is not minimized under a force field. A molecule that
/// already has 3D coordinates keeps the stereo they show instead of its
/// wedges. `observer` sees the embedder's minimizer and can cancel it.
pub fn build_3d(drawing: &Molecule, params: &EmbedParams, observer: &mut dyn Observer) -> Result<Molecule> {
    let mut mol = drawing.clone();
    let flat = !perception::has_3d_coords(&mol);
    hydrogens::add_hydrogens(&mut mol);
    if flat {
        stereo::assign_stereo_from_2d(&mut mol);
    } else {
        // the hydrogens are placed along the 3D geometry, so it still shows
        // the stereo with them in place
        stereo::assign_chiral_tags_from_3d(&mut mol);
        stereo::assign_double_bond_stereo_from_3d(&mut mol);
    }
    // The embedder must enforce these tags, not re-read the input's
    // (planar, or hydrogen-added) coordinates.
    let params = EmbedParams { rdkit_stereo_from_3d: false, ..params.clone() };
    mol.coords = distgeom::embed_molecule(&mol, &params, observer)?;
    Ok(mol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BondStereo, ChiralTag};
    use crate::optim::NoObserver;

    /// (R)/(S)-CHFClBr as an editor writes it: flat, H implicit, the Br bond wedged.
    fn chfclbr(wedge: u8) -> String {
        format!(
            "\n  editor          2D\n\n  4  3  0  0  0  0  0  0  0  0999 V2000\n\
             \x20   0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
             \x20   1.0000    0.0000    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0\n\
             \x20  -0.5000    0.8700    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0\n\
             \x20  -0.5000   -0.8700    0.0000 Br  0  0  0  0  0  0  0  0  0  0  0  0\n\
             \x20 1  2  1  0\n  1  3  1  0\n  1  4  1  {wedge}\nM  END\n"
        )
    }

    fn params() -> EmbedParams {
        EmbedParams { random_seed: 7, ..EmbedParams::etkdg_v3() }
    }

    #[test]
    fn the_wedge_decides_the_enantiomer_built() {
        let mut tags = Vec::new();
        for wedge in [1, 6] {
            let drawing = crate::io::read_molfile(&chfclbr(wedge)).unwrap();
            let built = build_3d(&drawing, &params(), &mut NoObserver).unwrap();
            assert_eq!(built.num_atoms(), 5);
            assert!(perception::has_3d_coords(&built));
            let want = built.atom(0).chiral_tag;
            assert_ne!(want, ChiralTag::Unspecified);
            let mut seen = built.clone();
            stereo::assign_chiral_tags_from_3d(&mut seen);
            assert_eq!(seen.atom(0).chiral_tag, want, "wedge {wedge}");
            tags.push(want);
        }
        assert_ne!(tags[0], tags[1]);
    }

    #[test]
    fn a_drawn_cis_double_bond_is_built_cis() {
        // cis-2-butene drawn flat
        let text = "\n  editor          2D\n\n  4  3  0  0  0  0  0  0  0  0999 V2000\n\
            \x20  -1.0000    0.8000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
            \x20  -0.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
            \x20   0.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
            \x20   1.0000    0.8000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n\
            \x20 1  2  1  0\n  2  3  2  0\n  3  4  1  0\nM  END\n";
        let drawing = crate::io::read_molfile(text).unwrap();
        let built = build_3d(&drawing, &params(), &mut NoObserver).unwrap();
        assert_eq!(built.num_atoms(), 12);
        let d = crate::core::geometry::dihedral(built.pos(0), built.pos(1), built.pos(2), built.pos(3)).to_degrees();
        assert!(d.abs() < 30.0, "dihedral {d}");
        assert_eq!(built.bond(1).stereo, BondStereo::Cis);
    }
}
