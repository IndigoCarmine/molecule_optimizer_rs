//! [`ConformerGenerator`] implementation backed by the ETKDG port.

use super::{EmbedParams, Embedding};
use crate::calc::conformer::ConformerGenerator;
use crate::core::Molecule;
use crate::core::geometry::V3;
use crate::error::{Error, Result};
use crate::optim::Observer;
use std::sync::{Arc, Mutex};

/// Generates conformer `conf_id` exactly like conformer `conf_id` of RDKit
/// `EmbedMultipleConfs(mol, N, params)` (same per-conformer seed), so a
/// conformer search with N starting geometries starts from RDKit's
/// geometries. Failed embeddings return an error (RDKit drops them).
#[derive(Debug)]
pub struct EtkdgGenerator {
    pub params: EmbedParams,
    cache: Mutex<Option<(u64, Arc<Embedding>)>>,
}

impl Clone for EtkdgGenerator {
    fn clone(&self) -> Self {
        EtkdgGenerator::new(self.params.clone())
    }
}

impl EtkdgGenerator {
    pub fn new(params: EmbedParams) -> Self {
        EtkdgGenerator { params, cache: Mutex::new(None) }
    }

    fn key(mol: &Molecule) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for a in mol.atoms() {
            (a.atomic_num, a.formal_charge, a.chiral_tag as u8).hash(&mut h);
        }
        for b in mol.bonds() {
            (b.begin, b.end, b.order as u8, b.stereo as u8, b.stereo_atoms).hash(&mut h);
        }
        for p in &mol.coords {
            for x in p {
                x.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }

    /// The prepared embedding for `mol` (cached for repeated calls).
    pub fn embedding(&self, mol: &Molecule) -> Result<Arc<Embedding>> {
        let k = Self::key(mol);
        let mut g = self.cache.lock().map_err(|_| Error::Job("poisoned cache".into()))?;
        if let Some((key, e)) = g.as_ref() {
            if *key == k {
                return Ok(e.clone());
            }
        }
        let e = Arc::new(Embedding::new(mol, &self.params)?);
        *g = Some((k, e.clone()));
        Ok(e)
    }
}

impl Default for EtkdgGenerator {
    fn default() -> Self {
        EtkdgGenerator::new(EmbedParams::etkdg_v3())
    }
}

impl ConformerGenerator for EtkdgGenerator {
    fn name(&self) -> &str {
        "ETKDG"
    }
    fn generate(&self, mol: &Molecule, conf_id: usize, observer: &mut dyn Observer) -> Result<Vec<V3>> {
        let e = self.embedding(mol)?;
        e.embed_conformer(conf_id, observer)?
            .ok_or_else(|| Error::Embedding(format!("embedding of conformer {conf_id} failed")))
    }
}
