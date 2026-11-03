//! Maximum cardinality matching in general graphs (Edmonds' blossom algorithm).

use std::collections::VecDeque;

/// Returns `mate[v]` (`usize::MAX` = unmatched) for a maximum matching of the
/// graph given by adjacency lists. `preferred` may contain an initial
/// (greedy) matching to speed things up.
pub fn max_matching(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    const NONE: usize = usize::MAX;
    let mut mate = vec![NONE; n];
    // greedy init
    for v in 0..n {
        if mate[v] == NONE {
            if let Some(&u) = adj[v].iter().find(|&&u| mate[u] == NONE && u != v) {
                mate[v] = u;
                mate[u] = v;
            }
        }
    }
    let mut p = vec![NONE; n];
    let mut base = vec![0usize; n];
    let mut used = vec![false; n];
    let mut blossom = vec![false; n];

    let lca = |mate: &[usize], base: &[usize], p: &[usize], mut a: usize, mut b: usize| -> usize {
        let mut used_l = vec![false; n];
        loop {
            a = base[a];
            used_l[a] = true;
            if mate[a] == NONE {
                break;
            }
            a = p[mate[a]];
        }
        loop {
            b = base[b];
            if used_l[b] {
                return b;
            }
            b = p[mate[b]];
        }
    };

    for root in 0..n {
        if mate[root] != NONE || adj[root].is_empty() {
            continue;
        }
        // find augmenting path from root
        used.iter_mut().for_each(|x| *x = false);
        p.iter_mut().for_each(|x| *x = NONE);
        for (i, b) in base.iter_mut().enumerate() {
            *b = i;
        }
        used[root] = true;
        let mut q = VecDeque::new();
        q.push_back(root);
        let mut found = NONE;
        'bfs: while let Some(v) = q.pop_front() {
            for &to in &adj[v] {
                if base[v] == base[to] || mate[v] == to {
                    continue;
                }
                if to == root || (mate[to] != NONE && p[mate[to]] != NONE) {
                    let curbase = lca(&mate, &base, &p, v, to);
                    blossom.iter_mut().for_each(|x| *x = false);
                    // mark path
                    let mark = |mut vv: usize, b: usize, mut child: usize, p: &mut Vec<usize>, blossom: &mut Vec<bool>, base: &Vec<usize>, mate: &Vec<usize>| {
                        while base[vv] != b {
                            blossom[base[vv]] = true;
                            blossom[base[mate[vv]]] = true;
                            p[vv] = child;
                            child = mate[vv];
                            vv = p[mate[vv]];
                        }
                    };
                    mark(v, curbase, to, &mut p, &mut blossom, &base, &mate);
                    mark(to, curbase, v, &mut p, &mut blossom, &base, &mate);
                    for i in 0..n {
                        if blossom[base[i]] {
                            base[i] = curbase;
                            if !used[i] {
                                used[i] = true;
                                q.push_back(i);
                            }
                        }
                    }
                } else if p[to] == NONE {
                    p[to] = v;
                    if mate[to] == NONE {
                        found = to;
                        break 'bfs;
                    }
                    let m = mate[to];
                    used[m] = true;
                    q.push_back(m);
                }
            }
        }
        if found != NONE {
            let mut v = found;
            while v != NONE {
                let pv = p[v];
                let ppv = mate[pv];
                mate[v] = pv;
                mate[pv] = v;
                v = ppv;
            }
        }
    }
    mate
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn odd_cycle_plus_tail() {
        // 5-cycle 0..4 plus pendant 5 on 0 → perfect matching of 6 vertices
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)];
        let mut adj = vec![Vec::new(); 6];
        for &(a, b) in &edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        let m = max_matching(&adj);
        assert!(m.iter().all(|&x| x != usize::MAX));
    }
}
