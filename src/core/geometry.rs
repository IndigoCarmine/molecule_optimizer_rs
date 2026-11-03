//! Small 3D vector helpers and internal-coordinate utilities.

pub type V3 = [f64; 3];

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
pub fn normalize(a: V3) -> V3 {
    let n = norm(a);
    if n > 0.0 { scale(a, 1.0 / n) } else { a }
}
#[inline]
pub fn dist(a: V3, b: V3) -> f64 {
    norm(sub(a, b))
}

/// Reads point `i` from a flat coordinate array with dimension `dim` (≥3).
#[inline]
pub fn pt(pos: &[f64], i: usize, dim: usize) -> V3 {
    [pos[dim * i], pos[dim * i + 1], pos[dim * i + 2]]
}

/// Angle a-b-c in radians.
pub fn angle(a: V3, b: V3, c: V3) -> f64 {
    let u = sub(a, b);
    let v = sub(c, b);
    let cs = dot(u, v) / (norm(u) * norm(v)).max(1e-16);
    cs.clamp(-1.0, 1.0).acos()
}

/// Dihedral a-b-c-d in radians, range (-π, π] (IUPAC sign convention).
pub fn dihedral(a: V3, b: V3, c: V3, d: V3) -> f64 {
    let b0 = sub(a, b);
    let b1 = sub(c, b);
    let b2 = sub(d, c);
    let b1n = normalize(b1);
    let v = sub(b0, scale(b1n, dot(b0, b1n)));
    let w = sub(b2, scale(b1n, dot(b2, b1n)));
    let x = dot(v, w);
    let y = dot(cross(b1n, v), w);
    y.atan2(x)
}

/// Rotates point `p` about the axis through `origin` with direction `axis`
/// by `theta` radians (Rodrigues' formula).
pub fn rotate_about_axis(p: V3, origin: V3, axis: V3, theta: f64) -> V3 {
    let k = normalize(axis);
    let v = sub(p, origin);
    let (s, c) = theta.sin_cos();
    let kv = cross(k, v);
    let kd = dot(k, v);
    let r = [
        v[0] * c + kv[0] * s + k[0] * kd * (1.0 - c),
        v[1] * c + kv[1] * s + k[1] * kd * (1.0 - c),
        v[2] * c + kv[2] * s + k[2] * kd * (1.0 - c),
    ];
    add(r, origin)
}

/// Optimal superposition RMSD (Kabsch) between two point sets of equal length.
/// Returns `(rmsd, rotation (row-major), translation)` such that
/// `R * (q_i) + t ≈ p_i`.
pub fn kabsch_rmsd(p: &[V3], q: &[V3]) -> f64 {
    kabsch(p, q).0
}

/// Kabsch alignment of `q` onto `p`. Returns `(rmsd, R, t)`.
pub fn kabsch(p: &[V3], q: &[V3]) -> (f64, [[f64; 3]; 3], V3) {
    use nalgebra::{Matrix3, Vector3};
    assert_eq!(p.len(), q.len());
    let n = p.len();
    if n == 0 {
        return (0.0, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0; 3]);
    }
    let cp = centroid(p);
    let cq = centroid(q);
    let mut h = Matrix3::<f64>::zeros();
    for i in 0..n {
        let a = Vector3::from(sub(q[i], cq));
        let b = Vector3::from(sub(p[i], cp));
        h += a * b.transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut d = Matrix3::<f64>::identity();
    if (vt.transpose() * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = vt.transpose() * d * u.transpose();
    let cqv = Vector3::from(cq);
    let cpv = Vector3::from(cp);
    let t = cpv - r * cqv;
    let mut ss = 0.0;
    for i in 0..n {
        let x = r * Vector3::from(q[i]) + t;
        let y = Vector3::from(p[i]);
        ss += (x - y).norm_squared();
    }
    let rm = [
        [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
        [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
        [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
    ];
    ((ss / n as f64).sqrt(), rm, [t[0], t[1], t[2]])
}

pub fn centroid(p: &[V3]) -> V3 {
    let mut c = [0.0; 3];
    for x in p {
        c = add(c, *x);
    }
    let n = p.len().max(1) as f64;
    scale(c, 1.0 / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dihedral_sign() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let c = [0.0, 0.0, 1.0];
        let d = [0.0, 1.0, 1.0];
        assert!((dihedral(a, b, c, d) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }
    #[test]
    fn kabsch_identity_rotation() {
        let p = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.5]];
        let q: Vec<V3> = p
            .iter()
            .map(|&x| add(rotate_about_axis(x, [0.0; 3], [1.0, 2.0, 3.0], 0.7), [3.0, -1.0, 2.0]))
            .collect();
        assert!(kabsch_rmsd(&p, &q) < 1e-9);
    }
}
