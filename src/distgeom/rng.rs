//! Random number generation exactly as used by RDKit's embedder.
//!
//! RDKit (`Code/RDGeneral/utils.h`) uses `boost::minstd_rand` wrapped in a
//! `boost::variate_generator` with `boost::uniform_real<>(0, 1)`. This module
//! reproduces that stream bit for bit so that a given `randomSeed` yields the
//! same distance matrices / coordinates as RDKit.
//!
//! Port of parts of RDKit `Code/RDGeneral/utils.{h,cpp}` and the boost.random
//! semantics it relies on. RDKit: Copyright (C) 2002-2026 Greg Landrum and
//! other RDKit contributors, BSD-3-Clause.

const MODULUS: u64 = 2_147_483_647;
const MULTIPLIER: u64 = 48_271;
const ENGINE_MIN: u32 = 1;
const ENGINE_MAX: u32 = 2_147_483_646;

/// `boost::random::minstd_rand` (Park–Miller, multiplier 48271).
#[derive(Debug, Clone)]
pub struct MinstdRand {
    x: u32,
}

impl MinstdRand {
    /// Equivalent to `rng_type gen(42u); gen.seed(seed);`.
    pub fn new(seed: u32) -> Self {
        let mut r = MinstdRand { x: 1 };
        r.seed(seed);
        r
    }

    /// boost `linear_congruential_engine::seed(IntType)` (increment 0).
    pub fn seed(&mut self, seed: u32) {
        let mut x = (seed as u64 % MODULUS) as u32;
        if x == 0 {
            x = 1;
        }
        self.x = x;
    }

    /// Next raw value in `[1, 2147483646]`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u32 {
        self.x = ((self.x as u64 * MULTIPLIER) % MODULUS) as u32;
        self.x
    }
}

/// `variate_generator<minstd_rand&, uniform_real<>>(gen, uniform_real(0,1))`:
/// a source of doubles in `[0, 1)`.
#[derive(Debug, Clone)]
pub struct DoubleSource {
    engine: MinstdRand,
}

impl DoubleSource {
    pub fn new(seed: u32) -> Self {
        DoubleSource { engine: MinstdRand::new(seed) }
    }

    /// boost `detail::generate_uniform_real` for an integral engine.
    pub fn next_f64(&mut self) -> f64 {
        loop {
            let numerator = (self.engine.next() - ENGINE_MIN) as f64;
            let divisor = (ENGINE_MAX - ENGINE_MIN) as f64 + 1.0;
            let result = numerator / divisor * (1.0 - 0.0) + 0.0;
            if result < 1.0 {
                return result;
            }
        }
    }
}

/// Seed used by RDKit when no seed is given (`randomSeed = -1`): RDKit uses a
/// global generator seeded with 42 that is shared between calls. We use a
/// fresh generator seeded from the system time instead.
pub fn unseeded_source() -> DoubleSource {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);
    DoubleSource::new(((t ^ (t >> 32)) as u32) | 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minstd_reference_sequence() {
        // std::minstd_rand default seed 1: 10000th value is 399268537 (C++ standard)
        let mut g = MinstdRand::new(1);
        let mut v = 0;
        for _ in 0..10000 {
            v = g.next();
        }
        assert_eq!(v, 399_268_537);
    }

    #[test]
    fn zero_seed_maps_to_one() {
        let mut a = MinstdRand::new(0);
        let mut b = MinstdRand::new(1);
        assert_eq!(a.next(), b.next());
    }

    #[test]
    fn uniform_in_range() {
        let mut s = DoubleSource::new(42);
        for _ in 0..1000 {
            let x = s.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
