//! Relevant cycles in RDKit's order: a port of RingDecomposerLib (the
//! library behind RDKit's default ring perception) and of the RDKit glue that
//! turns its relevant-cycle iterator into `RingInfo::AtomRings()`.
//!
//! Ported from
//! * RingDecomposerLib 1.1.3 (`src/RingDecomposerLib/RDLgraph.c`,
//!   `RDLtarjan.c`, `RDLapsp.c`, `RDLcycleFams.c`, `RDLrelation.c`,
//!   `RDLinfo.c`, `RDLhandler.c`, `RingDecomposerLib.c`),
//!   Copyright (c) 2016 University of Hamburg, ZBH - Center for
//!   Bioinformatics, Niek Andresen, Florian Flachsenberg, Matthias Rarey;
//!   BSD 3-Clause license.
//! * RDKit 2026.09.1 `Code/GraphMol/FindRings.cpp` (`symmetrizeSSSR`,
//!   `findRingFamilies`) and `Code/GraphMol/RingInfo.cpp`
//!   (`atomRelevantCycles`, `RingUtils::rdlCycleToAtomRing`,
//!   `RingUtils::normalizeRing`),
//!   Copyright (C) 2003-2026 Greg Landrum and other RDKit contributors;
//!   BSD 3-Clause license (see `LICENSE-RDKIT.txt`).
//!
//! RDKit 2026.09's `MolOps::sanitizeMol` calls `symmetrizeSSSR` with the
//! default algorithm, which (unless the `RDK_USE_LEGACY_RING_FINDING`
//! environment variable is set) stores RingDecomposerLib's relevant cycles,
//! in the order of `RDL_getRCyclesIterator`: per biconnected component (in
//! Tarjan completion order), per unique ring family, per relevant cycle
//! family, per pair of shortest paths.  Each ring is then normalized (start at
//! the smallest atom index, continue towards the smaller neighbor).
//!
//! RingDecomposerLib sorts the cycle families by weight; families of equal
//! weight keep their generation order (as in the reference RDKit build).

use crate::core::{BondOrder, Molecule};

/// Undirected graph with RDL's adjacency layout: `adj[v]` lists
/// `(neighbor, edge id)` in edge-insertion order; `edges[e]` is `[min, max]`.
struct Graph {
    adj: Vec<Vec<(usize, usize)>>,
    edges: Vec<[usize; 2]>,
}

impl Graph {
    fn new(v: usize) -> Graph {
        Graph { adj: vec![Vec::new(); v], edges: Vec::new() }
    }
    fn v(&self) -> usize {
        self.adj.len()
    }
    fn e(&self) -> usize {
        self.edges.len()
    }
    fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }
    /// `RDL_addUEdge_g`: loops and duplicate edges are ignored.
    fn add_uedge(&mut self, from: usize, to: usize) {
        if from == to || self.adj[from].iter().any(|&(n, _)| n == to) {
            return;
        }
        let id = self.edges.len();
        self.adj[from].push((to, id));
        self.adj[to].push((from, id));
        self.edges.push([from.min(to), from.max(to)]);
    }
    /// `RDL_edgeId`.
    fn edge_id(&self, a: usize, b: usize) -> usize {
        let (from, to) = (a.min(b), a.max(b));
        self.adj[from].iter().find(|&&(n, _)| n == to).map(|&(_, e)| e).expect("edge")
    }
}

/// Directed graph without edge ids (RDL's `dPaths` graphs, built with
/// `RDL_addEdge`).
struct DGraph {
    adj: Vec<Vec<usize>>,
}

impl DGraph {
    fn add_edge(&mut self, from: usize, to: usize) {
        if from == to || self.adj[from].contains(&to) {
            return;
        }
        self.adj[from].push(to);
    }
}

type Bits = Vec<u64>;

fn bits_new(n: usize) -> Bits {
    vec![0u64; n.div_ceil(64).max(1)]
}
fn bit_test(b: &[u64], i: usize) -> bool {
    b[i / 64] >> (i % 64) & 1 == 1
}
fn bit_set(b: &mut [u64], i: usize) {
    b[i / 64] |= 1u64 << (i % 64);
}
fn bits_xor(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= *s;
    }
}
fn bits_empty(b: &[u64]) -> bool {
    b.iter().all(|&w| w == 0)
}
/// `RDL_swap_columns`.
fn swap_columns(rows: &mut [Bits], c1: usize, c2: usize) {
    for r in rows.iter_mut() {
        let v1 = bit_test(r, c1);
        let v2 = bit_test(r, c2);
        if v1 != v2 {
            r[c1 / 64] ^= 1u64 << (c1 % 64);
            r[c2 / 64] ^= 1u64 << (c2 % 64);
        }
    }
}

// ---------------------------------------------------------------------------
// Tarjan's biconnected components (RDLtarjan.c)
// ---------------------------------------------------------------------------

struct BccGraph {
    graphs: Vec<Graph>,
    /// local edge index -> global edge index, per BCC
    edge_from: Vec<Vec<usize>>,
}

/// `RDL_tarjan`: 1-based BCC label for each edge.
fn tarjan(g: &Graph) -> (Vec<usize>, usize) {
    let n = g.v();
    let mut d = vec![0usize; n];
    let mut low = vec![0usize; n];
    let mut bcc = vec![0usize; g.e()];
    let mut time = 0usize;
    let mut curr_bcc = 1usize;
    let mut edge_stack: Vec<usize> = Vec::new();

    struct El {
        u: usize,
        parent: usize,
        j: usize,
        expected_time: usize,
    }

    for start in 0..n {
        if d[start] != 0 {
            continue;
        }
        let mut dfs: Vec<El> = Vec::new();
        time += 1;
        d[start] = time;
        low[start] = time;
        dfs.push(El { u: start, parent: usize::MAX, j: 0, expected_time: usize::MAX });
        while let Some(top) = dfs.last_mut() {
            let u = top.u;
            let parent = top.parent;
            let j = top.j;
            if j < g.degree(u) {
                let v = g.adj[u][j].0;
                let uv = g.edge_id(u, v);
                if d[v] == 0 {
                    edge_stack.push(uv);
                    time += 1;
                    d[v] = time;
                    low[v] = time;
                    top.expected_time = time;
                    dfs.push(El { u: v, parent: u, j: 0, expected_time: usize::MAX });
                } else {
                    top.j += 1;
                    if d[v] == top.expected_time {
                        low[u] = low[u].min(low[v]);
                        if low[v] >= d[u] {
                            loop {
                                let e = edge_stack.pop().expect("edge stack");
                                bcc[e] = curr_bcc;
                                if e == uv {
                                    break;
                                }
                            }
                            curr_bcc += 1;
                        }
                    } else if d[v] < d[u] && v != parent {
                        edge_stack.push(uv);
                        low[u] = low[u].min(d[v]);
                    }
                }
            } else {
                dfs.pop();
            }
        }
    }
    (bcc, curr_bcc - 1)
}

/// `RDL_tarjanBCC`: the non-trivial (more than one edge) biconnected
/// components as separate graphs with RDL's local node/edge numbering.
fn tarjan_bcc(g: &Graph) -> BccGraph {
    let (bcc, nof_bcc) = tarjan(g);
    let mut bcc_size = vec![0usize; nof_bcc];
    for &b in &bcc {
        bcc_size[b - 1] += 1;
    }
    let mut mapping = vec![usize::MAX; nof_bcc];
    let mut nontrivial = 0;
    for i in 0..nof_bcc {
        if bcc_size[i] > 1 {
            mapping[i] = nontrivial;
            nontrivial += 1;
        }
    }
    let mut edge_to: Vec<usize> = vec![usize::MAX; g.e()];
    let mut edge_from: Vec<Vec<usize>> = vec![Vec::new(); nontrivial];
    let mut node_to: Vec<Vec<(usize, usize)>> = vec![Vec::new(); g.v()];
    let mut nof_nodes = vec![0usize; nontrivial];
    for i in 0..g.e() {
        if bcc_size[bcc[i] - 1] <= 1 {
            continue;
        }
        let u = mapping[bcc[i] - 1];
        edge_to[i] = u;
        edge_from[u].push(i);
        for &node in &g.edges[i] {
            if !node_to[node].iter().any(|&(b, _)| b == u) {
                node_to[node].push((u, nof_nodes[u]));
                nof_nodes[u] += 1;
            }
        }
    }
    let mut graphs: Vec<Graph> = nof_nodes.iter().map(|&n| Graph::new(n)).collect();
    for i in 0..g.e() {
        let u = edge_to[i];
        if u == usize::MAX {
            continue;
        }
        let local = |node: usize| node_to[node].iter().find(|&&(b, _)| b == u).map(|&(_, l)| l).expect("bcc node");
        let (a, b) = (local(g.edges[i][0]), local(g.edges[i][1]));
        graphs[u].add_uedge(a, b);
    }
    BccGraph { graphs, edge_from }
}

// ---------------------------------------------------------------------------
// All pairs shortest paths (RDLapsp.c)
// ---------------------------------------------------------------------------

struct SPathInfo {
    pred: Vec<Vec<usize>>,
    dist: Vec<Vec<usize>>,
    reachable: Vec<Vec<bool>>,
    d_paths: Vec<DGraph>,
}

fn all_pairs_shortest_paths(g: &Graph) -> SPathInfo {
    const INF: usize = usize::MAX;
    let n = g.v();
    let mut spi = SPathInfo {
        pred: vec![vec![INF; n]; n],
        dist: vec![vec![INF; n]; n],
        reachable: vec![vec![false; n]; n],
        d_paths: (0..n).map(|_| DGraph { adj: vec![Vec::new(); n] }).collect(),
    };
    let mut color = vec![false; n];
    let mut queue: Vec<usize> = Vec::with_capacity(n);
    for run in 1..3 {
        for i in 0..n {
            queue.clear();
            for j in 0..n {
                if run == 1 {
                    spi.dist[i][j] = INF;
                    spi.pred[i][j] = INF;
                    spi.reachable[i][j] = false;
                }
                color[j] = false;
            }
            spi.dist[i][i] = 0;
            spi.pred[i][i] = i;
            color[i] = true;
            queue.push(i);
            let mut head = 0;
            while head < queue.len() {
                let w = queue[head];
                head += 1;
                for &(j, _) in &g.adj[w] {
                    let precedes =
                        run == 2 || (g.degree(j) == g.degree(i) && j < i) || g.degree(j) < g.degree(i);
                    if precedes && !color[j] {
                        let dw1 = spi.dist[i][w].wrapping_add(1);
                        if run == 2 && dw1 < spi.dist[i][j] {
                            spi.reachable[i][j] = false;
                        }
                        if run == 1 || dw1 < spi.dist[i][j] {
                            spi.pred[i][j] = w;
                        }
                        if spi.dist[i][j] > dw1 {
                            spi.dist[i][j] = dw1;
                        }
                        color[j] = true;
                        queue.push(j);
                        if run == 1 {
                            spi.reachable[i][j] = true;
                        }
                    }
                }
            }
        }
    }
    spi
}

// ---------------------------------------------------------------------------
// Cycle families, Vismara's algorithm (RDLcycleFams.c)
// ---------------------------------------------------------------------------

struct CFam {
    weight: usize,
    r: usize,
    p: usize,
    q: usize,
    /// `usize::MAX` for odd families
    x: usize,
    prototype: Vec<bool>,
    mark: bool,
}

fn paths_share_only_start(r: usize, y: usize, z: usize, g: &Graph, spi: &SPathInfo) -> bool {
    let n = g.v();
    let mut in_ry = vec![false; n];
    let mut in_rz = vec![false; n];
    in_ry[y] = true;
    in_rz[z] = true;
    let mut pnt = y;
    loop {
        pnt = spi.pred[r][pnt];
        in_ry[pnt] = true;
        if pnt == r {
            break;
        }
    }
    pnt = z;
    loop {
        pnt = spi.pred[r][pnt];
        in_rz[pnt] = true;
        if pnt == r {
            break;
        }
    }
    let mut count = 0;
    for i in 0..n {
        if count >= 2 {
            break;
        }
        if in_ry[i] && in_rz[i] {
            count += 1;
        }
    }
    count == 1 && in_ry[r] && in_rz[r]
}

fn find_prototype(r: usize, y: usize, z: usize, x: usize, g: &Graph, spi: &SPathInfo) -> Vec<bool> {
    let mut proto = vec![false; g.e()];
    for start in [y, z] {
        let mut v1 = start;
        loop {
            let v2 = v1;
            v1 = spi.pred[r][v1];
            proto[g.edge_id(v1, v2)] = true;
            if v1 == r {
                break;
            }
        }
    }
    if x == usize::MAX {
        proto[g.edge_id(y, z)] = true;
    } else {
        proto[g.edge_id(y, x)] = true;
        proto[g.edge_id(z, x)] = true;
    }
    proto
}

fn vismara(g: &Graph, spi: &mut SPathInfo) -> Vec<CFam> {
    let n = g.v();
    let mut fams = Vec::new();
    let mut even_cand: Vec<usize> = Vec::with_capacity(n);
    for rv in 0..n {
        for yv in 0..n {
            if !spi.reachable[rv][yv] {
                continue;
            }
            even_cand.clear();
            for &(zv, _) in &g.adj[yv] {
                if !spi.reachable[rv][zv] {
                    continue;
                }
                if spi.dist[rv][zv] + 1 == spi.dist[rv][yv] {
                    even_cand.push(zv);
                } else if spi.dist[rv][zv] != spi.dist[rv][yv] + 1
                    && (g.degree(zv) < g.degree(yv) || (g.degree(zv) == g.degree(yv) && zv < yv))
                    && paths_share_only_start(rv, yv, zv, g, spi)
                {
                    fams.push(CFam {
                        weight: spi.dist[rv][yv] + spi.dist[rv][zv] + 1,
                        r: rv,
                        p: yv,
                        q: zv,
                        x: usize::MAX,
                        prototype: find_prototype(rv, yv, zv, usize::MAX, g, spi),
                        mark: false,
                    });
                }
                if spi.dist[rv][zv] == 1 {
                    spi.d_paths[rv].add_edge(zv, rv);
                }
            }
            for i in 0..even_cand.len() {
                let pv = even_cand[i];
                for &qv in &even_cand[i + 1..] {
                    if paths_share_only_start(rv, pv, qv, g, spi) {
                        fams.push(CFam {
                            weight: spi.dist[rv][pv] + spi.dist[rv][qv] + 2,
                            r: rv,
                            p: pv,
                            q: qv,
                            x: yv,
                            prototype: find_prototype(rv, pv, qv, yv, g, spi),
                            mark: false,
                        });
                    }
                }
            }
            for &pv in &even_cand {
                spi.d_paths[rv].add_edge(yv, pv);
            }
        }
    }
    fams
}

/// `RDL_findCycleFams`.
fn find_cycle_fams(g: &Graph, spi: &mut SPathInfo) -> Vec<CFam> {
    let mut fams = vismara(g, spi);
    // RDL sorts the families by weight with `qsort`; the reference RDKit
    // build keeps ties in generation order, i.e. behaves as a stable sort
    // (verified on all fixtures of tests/ring_reference.rs).
    fams.sort_by_key(|f| f.weight);
    fams
}

// ---------------------------------------------------------------------------
// URF relation (RDLrelation.c, RDLinfo.c)
// ---------------------------------------------------------------------------

struct UrfInfo {
    nof_protos: Vec<usize>,
    /// per weight: matrix of URF relations between its families
    urf_rel: Vec<Vec<Vec<bool>>>,
    /// URFs as lists of family indices (into the sorted family array)
    urfs: Vec<Vec<usize>>,
}

impl UrfInfo {
    fn idx(&self, weight: usize, j: usize) -> usize {
        self.nof_protos[..weight].iter().sum::<usize>() + j
    }
}

fn check_dependencies(fams: &mut [CFam], g: &Graph, u: &mut UrfInfo) {
    let nof_weights = u.nof_protos.len();
    if fams.len() < 3 {
        for i in 0..nof_weights {
            for j in 0..u.nof_protos[i] {
                u.urf_rel[i][j][j] = true;
            }
        }
        for f in fams.iter_mut() {
            f.mark = true;
        }
        return;
    }
    let ne = g.e();
    let mut protos: Vec<Bits> = fams
        .iter()
        .map(|f| {
            let mut b = bits_new(ne);
            for (k, &on) in f.prototype.iter().enumerate() {
                if on {
                    bit_set(&mut b, k);
                }
            }
            b
        })
        .collect();
    let mut basis: Vec<Bits> = Vec::new();
    let mut relevant: Vec<Bits> = Vec::new();
    let mut relevant_map: Vec<usize> = Vec::new();
    for i in 0..nof_weights {
        let basis_smaller_end = basis.len();
        let relevant_smaller_end = relevant.len();
        for j in 0..u.nof_protos[i] {
            let index = u.idx(i, j);
            let mut with_smaller = protos[index].clone();
            for k in 0..basis_smaller_end {
                if bit_test(&with_smaller, k) {
                    bits_xor(&mut with_smaller, &basis[k]);
                }
            }
            if bits_empty(&with_smaller) {
                continue;
            }
            relevant.push(with_smaller.clone());
            relevant_map.push(j);
            fams[index].mark = true;
            u.urf_rel[i][j][j] = true;

            let mut with_equal = with_smaller.clone();
            for k in basis_smaller_end..basis.len() {
                if bit_test(&with_equal, k) {
                    bits_xor(&mut with_equal, &basis[k]);
                }
            }
            if bits_empty(&with_equal) {
                for k in relevant_smaller_end..relevant.len() - 1 {
                    let mut t = with_smaller.clone();
                    bits_xor(&mut t, &relevant[k]);
                    if bits_empty(&t) {
                        u.urf_rel[i][j][relevant_map[k]] = true;
                        u.urf_rel[i][relevant_map[k]][j] = true;
                    }
                }
            } else {
                let old = basis.len();
                basis.push(with_equal);
                if !bit_test(&basis[old], old) {
                    for k in old + 1..ne {
                        if bit_test(&basis[old], k) {
                            swap_columns(&mut basis, old, k);
                            swap_columns(&mut relevant, old, k);
                            swap_columns(&mut protos, old, k);
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// `RDL_giveEdges`.
fn give_edges(a: usize, b: usize, out: &mut [bool], g: &Graph, spi: &SPathInfo, visited: &mut [bool]) {
    if a == b {
        return;
    }
    visited[b] = true;
    for &vertex in &spi.d_paths[a].adj[b] {
        out[g.edge_id(b, vertex)] = true;
        if !visited[vertex] {
            give_edges(a, vertex, out, g, spi, visited);
        }
    }
}

/// `RDL_findEdges`: all edges of all cycles of a family.
fn find_edges(f: &CFam, g: &Graph, spi: &SPathInfo) -> Vec<bool> {
    let mut edges = vec![false; g.e()];
    let mut visited = vec![false; g.v().max(g.e())];
    give_edges(f.r, f.p, &mut edges, g, spi, &mut visited);
    visited.iter_mut().for_each(|v| *v = false);
    give_edges(f.r, f.q, &mut edges, g, spi, &mut visited);
    if f.x != usize::MAX {
        edges[g.edge_id(f.x, f.p)] = true;
        edges[g.edge_id(f.x, f.q)] = true;
    } else {
        edges[g.edge_id(f.p, f.q)] = true;
    }
    edges
}

fn check_edges(fams: &[CFam], g: &Graph, u: &mut UrfInfo, spi: &SPathInfo) {
    for i in 0..u.nof_protos.len() {
        let np = u.nof_protos[i];
        let mut edge_sets: Vec<Option<Vec<bool>>> = vec![None; np];
        for j in 0..np {
            for k in j + 1..np {
                if !u.urf_rel[i][j][k] {
                    continue;
                }
                for t in [j, k] {
                    if edge_sets[t].is_none() {
                        edge_sets[t] = Some(find_edges(&fams[u.idx(i, t)], g, spi));
                    }
                }
                let ej = edge_sets[j].as_ref().unwrap();
                let ek = edge_sets[k].as_ref().unwrap();
                let shared = ej.iter().zip(ek).any(|(&a, &b)| a && b);
                u.urf_rel[i][j][k] = shared;
                u.urf_rel[i][k][j] = shared;
            }
        }
    }
}

fn find_transitive_closure(u: &mut UrfInfo) {
    for i in 0..u.nof_protos.len() {
        let np = u.nof_protos[i];
        let mut visited = vec![false; np];
        let mut ccs: Vec<Vec<usize>> = Vec::new();
        for j in 0..np {
            if visited[j] {
                continue;
            }
            let mut cc = Vec::new();
            let mut stack = vec![j];
            visited[j] = true;
            while let Some(k) = stack.pop() {
                cc.push(k);
                for l in 0..np {
                    if !visited[l] && u.urf_rel[i][k][l] {
                        visited[l] = true;
                        stack.push(l);
                    }
                }
            }
            ccs.push(cc);
        }
        for cc in &ccs {
            for k in 0..cc.len() {
                for l in k + 1..cc.len() {
                    u.urf_rel[i][cc[k]][cc[l]] = true;
                    u.urf_rel[i][cc[l]][cc[k]] = true;
                }
            }
        }
    }
}

/// `RDL_checkURFRelation` (`RDL_initUrfInfo`, `RDL_findRelations`,
/// `RDL_fillURFs`).
fn check_urf_relation(fams: &mut [CFam], g: &Graph, spi: &SPathInfo) -> UrfInfo {
    let mut nof_protos: Vec<usize> = Vec::new();
    let mut curr = fams[0].weight;
    nof_protos.push(0);
    for f in fams.iter() {
        if f.weight != curr {
            nof_protos.push(0);
            curr = f.weight;
        }
        *nof_protos.last_mut().unwrap() += 1;
    }
    let urf_rel = nof_protos.iter().map(|&n| vec![vec![false; n]; n]).collect();
    let mut u = UrfInfo { nof_protos, urf_rel, urfs: Vec::new() };
    check_dependencies(fams, g, &mut u);
    check_edges(fams, g, &mut u, spi);
    find_transitive_closure(&mut u);
    // RDL_fillURFs
    for i in 0..u.nof_protos.len() {
        let np = u.nof_protos[i];
        let mut already = vec![false; np];
        for j in 0..np {
            if u.urf_rel[i][j][j] && !already[j] {
                let mut urf = Vec::new();
                for k in j..np {
                    if u.urf_rel[i][j][k] {
                        urf.push(u.idx(i, k));
                        already[k] = true;
                    }
                }
                u.urfs.push(urf);
            }
        }
    }
    u
}

// ---------------------------------------------------------------------------
// Relevant cycle enumeration (RDLhandler.c)
// ---------------------------------------------------------------------------

/// All shortest paths from `p` back to `r` in `U_r`, as edge bitsets, in the
/// order of RDL's path iterator (`RDL_listPaths`, mode 'b').
fn list_paths(r: usize, p: usize, g: &Graph, spi: &SPathInfo) -> Vec<Bits> {
    fn rec(r: usize, v: usize, g: &Graph, spi: &SPathInfo, cur: &mut Bits, out: &mut Vec<Bits>) {
        if v == r {
            out.push(cur.clone());
            return;
        }
        for &w in &spi.d_paths[r].adj[v] {
            let saved = cur.clone();
            bit_set(cur, g.edge_id(v, w));
            rec(r, w, g, spi, cur, out);
            *cur = saved;
        }
    }
    let mut out = Vec::new();
    let mut cur = bits_new(g.e());
    rec(r, p, g, spi, &mut cur, &mut out);
    out
}

/// `RingUtils::normalizeRing`: rotate to the smallest index, then orient
/// towards the smaller neighbor.
fn normalize_ring(ring: &mut [usize]) {
    let start = ring.iter().enumerate().min_by_key(|&(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
    ring.rotate_left(start);
    if ring.len() > 2 && ring[ring.len() - 1] < ring[1] {
        ring[1..].reverse();
    }
}

/// Atom cycle from a list of (global) edges forming a simple cycle
/// (`RingUtils::rdlCycleToAtomRing`).
fn cycle_to_atom_ring(edges: &[[usize; 2]]) -> Vec<usize> {
    let w = edges.len();
    let mut ring = vec![edges[0][0], edges[0][1]];
    let mut unseen = vec![true; w];
    unseen[0] = false;
    while ring.len() < w {
        let before = ring.len();
        for e in 0..w {
            if !unseen[e] {
                continue;
            }
            let edge = edges[e];
            for jj in 0..2 {
                if *ring.last().unwrap() == edge[jj] {
                    ring.push(edge[1 - jj]);
                    unseen[e] = false;
                    break;
                }
            }
            if ring.len() == w {
                break;
            }
        }
        if ring.len() == before {
            break; // not a simple cycle (cannot happen for relevant cycles)
        }
    }
    normalize_ring(&mut ring);
    ring
}

/// RDKit's ring set and order (`symmetrizeSSSR` with the default RDL
/// algorithm): normalized atom rings.
pub fn rdkit_atom_rings(mol: &Molecule) -> Vec<Vec<usize>> {
    rdkit_atom_rings_with(mol, false)
}

/// [`rdkit_atom_rings`] with RDKit's `includeDativeBonds` option of
/// `symmetrizeSSSR` (dative bonds count as ring bonds when `true`).
pub fn rdkit_atom_rings_with(mol: &Molecule, include_dative: bool) -> Vec<Vec<usize>> {
    // findRingFamilies: zero-order (and, by default, dative) bonds are not
    // ring candidates
    let mut graph = Graph::new(mol.num_atoms().max(1));
    for b in 0..mol.num_bonds() {
        let bond = mol.bond(b);
        if bond.order == BondOrder::Zero || (bond.order == BondOrder::Dative && !include_dative) {
            continue;
        }
        graph.add_uedge(bond.begin, bond.end);
    }
    let bccs = tarjan_bcc(&graph);
    let mut rings = Vec::new();
    for (bi, g) in bccs.graphs.iter().enumerate() {
        let mut spi = all_pairs_shortest_paths(g);
        let mut fams = find_cycle_fams(g, &mut spi);
        if fams.is_empty() {
            continue;
        }
        let uinfo = check_urf_relation(&mut fams, g, &spi);
        for urf in &uinfo.urfs {
            for &fi in urf {
                let f = &fams[fi];
                let p1 = list_paths(f.r, f.p, g, &spi);
                let p2 = list_paths(f.r, f.q, g, &spi);
                for a in &p1 {
                    for b in &p2 {
                        let mut cyc = a.clone();
                        for (x, y) in cyc.iter_mut().zip(b) {
                            *x |= *y;
                        }
                        if f.x != usize::MAX {
                            bit_set(&mut cyc, g.edge_id(f.p, f.x));
                            bit_set(&mut cyc, g.edge_id(f.q, f.x));
                        } else {
                            bit_set(&mut cyc, g.edge_id(f.p, f.q));
                        }
                        let edges: Vec<[usize; 2]> = (0..g.e())
                            .filter(|&k| bit_test(&cyc, k))
                            .map(|k| graph.edges[bccs.edge_from[bi][k]])
                            .collect();
                        rings.push(cycle_to_atom_ring(&edges));
                    }
                }
            }
        }
    }
    rings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize() {
        let mut r = vec![5, 3, 9, 1, 7];
        normalize_ring(&mut r);
        assert_eq!(r, vec![1, 7, 5, 3, 9]);
        let mut r = vec![4, 2, 8, 6];
        normalize_ring(&mut r);
        assert_eq!(r, vec![2, 4, 6, 8]);
    }
}
