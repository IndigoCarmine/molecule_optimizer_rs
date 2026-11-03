# molecule_optimizer_rs

Pure-Rust molecular mechanics: single-point energies, geometry optimization, conformer search (ETKDG), coordinate scans and assembly search, run as async/parallel jobs that can be observed and cancelled while they run.

The force fields (UFF, MMFF94/MMFF94s), the BFGS minimizer and the distance-geometry embedder (ETKDG) are ports of the [RDKit](https://github.com/rdkit/rdkit) implementations and are tested against RDKit reference data.

## Features

- **Force fields**: UFF and MMFF94 / MMFF94s with analytic gradients, energy breakdown by term, distance / angle / dihedral / position constraints, and user-defined terms (`ff::custom`).
- **Fast exact path**: `FastBuilder` compiles a force field into SIMD-friendly tables (AVX selected at runtime), giving the same energies as the reference path at lower cost; an optional approximate nonbonded mode (cutoffs, reaction field / PME) is available for large systems, with an optional GPU backend (`gpu` feature, wgpu).
- **Minimization**: RDKit's BFGS (line search, gradient scaling and convergence criteria), L-BFGS for large systems.
- **Conformers**: ETKDG (v1/v2/v3) embedding, conformer search with pruning, relaxed coordinate scans.
- **Async jobs**: every calculation runs on a dedicated rayon pool through `tasks::Engine`; a `JobHandle` gives snapshots, progress, cancellation and a `Stream` of intermediate structures.
- **I/O**: SDF / MOL, MOL2 and XYZ input (bond orders and charges are inferred for coordinate-only input), SDF / XYZ output, atom typing and charge assignment independent of the file format.

## RDKit parity

The test suite compares against data generated with RDKit (`tools/gen_*.py`): atom types, partial charges, energies and per-term energies, gradients, ETKDG embeddings, ring perception, SMARTS matching and minimized structures. On the benchmark set (`benches/data/set`, 110 molecules), energies at the initial and at RDKit-optimized structures match RDKit to better than 1e-9 relative, and minimizations from the same start agree within 3e-7 kcal/mol and 1.2e-4 Å heavy-atom RMSD.

## Usage

Add the crate (not on crates.io yet):

```toml
[dependencies]
molecule_optimizer_rs = { git = "https://github.com/IndigoCarmine/molecule_optimizer_rs", tag = "v0.1.0" }
```

### Single-point energy

```rust
use molecule_optimizer_rs::{calc, ff::mmff::MmffBuilder, io};

fn main() -> molecule_optimizer_rs::Result<()> {
    let mol = io::read_file("examples/data/paracetamol.mol")?.remove(0);
    let e = calc::energy::energy(&mol, &MmffBuilder::default())?;
    println!("E = {:.6} kcal/mol", e.total);
    for (term, v) in &e.breakdown {
        println!("  {term:<14} {v:>12.6}");
    }
    Ok(())
}
```

### Geometry optimization

Synchronous:

```rust
use molecule_optimizer_rs::calc::{self, OptimizeOptions};
use molecule_optimizer_rs::ff::uff::UffBuilder;

let r = calc::optimize::optimize(&mol, &UffBuilder::default(), &OptimizeOptions::default())?;
println!("{:?} after {} iterations, E = {:.6}", r.status, r.iterations, r.energy);
```

Many structures in parallel:

```rust
use molecule_optimizer_rs::tasks::Engine;
use std::sync::Arc;

let items = mols.into_iter().map(|m| (Arc::new(m), None)).collect();
let results = Engine::global().optimize_batch(items, Arc::new(UffBuilder::default()), OptimizeOptions::default()).await?;
```

### Streaming intermediate structures

`JobHandle::stream()` (any job) and `frames()` (optimizations) yield the latest state as the calculation runs; a slow consumer skips intermediate states instead of slowing the calculation down. The stream ends with the final state.

```rust
use futures::StreamExt;

let job = Engine::global().optimize(mol, Arc::new(UffBuilder::default()), OptimizeOptions::default());
let mut frames = job.frames();
while let Some(f) = frames.next().await {
    println!("iteration {} E = {:.6}", f.iteration, f.energy); // f.coords: current structure
}
let result = job.await?;
```

Frames are published every `EngineConfig::publish_interval` (default 50 ms). For every iteration, use the synchronous API with an `optim::Observer`.

### Endless minimization

With `force_tol <= 0` an optimization never reports convergence and keeps iterating (ignoring `max_iters`) until it is cancelled — useful for live, interactive relaxation:

```rust
let mut opts = OptimizeOptions::default();
opts.minimize.force_tol = 0.0;
let job = Engine::global().optimize(mol, Arc::new(UffBuilder::default()), opts);
// ... consume job.frames(), then
job.cancel();
let r = job.await?; // r.status == MinimizeStatus::Cancelled
```

Endless mode applies to `optimize` / `optimize_batch` / `optim::minimize`; scans, conformer and assembly searches always terminate.

### Fast path

```rust
use molecule_optimizer_rs::fast::{FastBuilder, FastOptions};
use molecule_optimizer_rs::ff::mmff::MmffBuilder;

let builder = Arc::new(FastBuilder::new(MmffBuilder::default(), FastOptions::exact()));
// use it anywhere a force-field builder is accepted
```

### More examples

```sh
cargo run --release --example energy -- examples/data/paracetamol.mol
cargo run --release --example optimize -- input.sdf optimized.sdf
cargo run --release --example stream_optimize -- examples/data/paracetamol.mol endless
cargo run --release --example conformer_search -- examples/data/paracetamol.mol 100
cargo run --release --example scan
cargo run --release --example custom_terms
cargo run --release --example assembly_search
```

## Performance

`cargo bench --bench set` (110 drug-like molecules, 7–76 atoms, minimized from ETKDG structures to convergence, single thread per molecule):

| Force field | Reference path | Fast exact path |
|---|---|---|
| MMFF94 | ~300 ms | ~175 ms |
| UFF | ~220 ms | ~115 ms |

Other benchmarks: `cargo bench --bench fast` (energy + gradient of up to ~2000-residue peptides, batch optimization of 1000 structures), `cargo bench --bench approx`, `cargo bench --bench gpu --features gpu`.

On x86-64, SIMD kernels select AVX at runtime; `RUSTFLAGS="-C target-cpu=native"` can help further. Results of the minimizer can differ at rounding level between CPUs with and without FMA.

## Testing

```sh
cargo test --release
```

## License

BSD 3-Clause (see `LICENSE`). Large parts are ports of RDKit, also BSD 3-Clause; see `LICENSE-RDKIT.txt`.
