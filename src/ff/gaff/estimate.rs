//! Empirical rules of Wang et al., J. Comput. Chem. 25, 1157 (2004) for
//! force constants that are not in the parameter table. The angle rule is
//! eq. (5) as corrected in the erratum, J. Comput. Chem. 26, 114 (2005).

/// Elements of Tables 3 and 4, in table order.
const ELEMENTS: [u8; 10] = [1, 6, 7, 8, 9, 17, 35, 53, 15, 16];

fn slot(z: u8) -> Option<usize> {
    ELEMENTS.iter().position(|&e| e == z)
}

/// Table 3: `(i, j, r_ij^ref, ln K_ij)` with `i`, `j` indices into
/// [`ELEMENTS`].
const TABLE3: [(usize, usize, f64, f64); 52] = [
    (0, 0, 0.738, 4.661),
    (1, 1, 1.526, 7.643),
    (2, 2, 1.441, 7.634),
    (3, 3, 1.460, 7.561),
    (4, 4, 1.406, 7.358),
    (5, 5, 2.031, 8.648),
    (6, 6, 2.337, 9.012),
    (7, 7, 2.836, 9.511),
    (8, 8, 2.324, 8.805),
    (9, 9, 2.038, 8.316),
    (0, 1, 1.090, 6.217),
    (0, 2, 1.010, 6.057),
    (0, 3, 0.960, 5.794),
    (0, 4, 0.920, 5.600),
    (0, 5, 1.280, 6.937),
    (0, 6, 1.410, 7.301),
    (0, 7, 1.600, 7.802),
    (0, 8, 1.410, 7.257),
    (0, 9, 1.340, 7.018),
    (1, 2, 1.470, 7.504),
    (1, 3, 1.440, 7.347),
    (1, 4, 1.370, 7.227),
    (1, 5, 1.800, 8.241),
    (1, 6, 1.940, 8.478),
    (1, 7, 2.160, 8.859),
    (1, 8, 1.830, 8.237),
    (1, 9, 1.820, 8.117),
    (2, 3, 1.420, 7.526),
    (2, 4, 1.420, 7.475),
    (2, 5, 1.750, 8.266),
    (2, 6, 1.930, 8.593),
    (2, 7, 2.120, 8.963),
    (2, 8, 1.720, 8.212),
    (2, 9, 1.690, 8.073),
    (3, 4, 1.410, 7.375),
    (3, 5, 1.700, 8.097),
    (3, 6, 1.790, 8.276),
    (3, 7, 2.110, 8.854),
    (3, 8, 1.640, 7.957),
    (3, 9, 1.650, 7.922),
    (4, 5, 1.648, 7.947),
    (5, 7, 2.550, 9.309),
    (6, 7, 2.671, 9.380),
    (4, 8, 1.500, 7.592),
    (4, 9, 1.580, 7.733),
    (5, 8, 2.040, 8.656),
    (5, 9, 2.030, 8.619),
    (6, 8, 2.240, 8.729),
    (6, 9, 2.210, 8.728),
    (7, 8, 2.490, 9.058),
    (7, 9, 2.560, 9.161),
    (8, 9, 2.120, 8.465),
];

/// Table 4: `Z` per element of [`ELEMENTS`].
const TABLE4_Z: [f64; 10] = [0.784, 1.183, 1.212, 1.219, 1.166, 1.272, 1.378, 1.398, 1.620, 1.280];

/// Table 4: `C` of the central element (C, N, O, P, S only).
fn table4_c(z: u8) -> Option<f64> {
    match z {
        6 => Some(1.339),
        7 => Some(1.300),
        8 => Some(1.249),
        15 => Some(0.906),
        16 => Some(1.448),
        _ => None,
    }
}

fn table3(i: usize, j: usize) -> Option<(f64, f64)> {
    let (i, j) = (i.min(j), i.max(j));
    TABLE3.iter().find(|t| t.0 == i && t.1 == j).map(|t| (t.2, t.3))
}

/// Exponent `m` of eq. (3).
pub const BOND_POWER: f64 = 4.5;

/// Bond force constant `K_r = K_ij / r^4.5` (eq. 3) for a bond of length
/// `r` (Å) between the elements `z1` and `z2`. `K_ij` is taken from Table 3,
/// or from eq. (4) (with `r` as the reference length of the pair) for the
/// element pairs that the table does not list. `None` for elements outside
/// H, C, N, O, F, Cl, Br, I, P, S.
pub fn bond_force_constant(z1: u8, z2: u8, r: f64) -> Option<f64> {
    let (i, j) = (slot(z1)?, slot(z2)?);
    let kij = match table3(i, j) {
        Some((_, ln_k)) => ln_k.exp(),
        None => {
            let (rii, ln_kii) = table3(i, i)?;
            let (rjj, ln_kjj) = table3(j, j)?;
            let (wi, wj) = ((r - rjj).abs(), (r - rii).abs());
            if wi + wj == 0.0 { 0.5 * (ln_kii.exp() + ln_kjj.exp()) } else { (ln_kii.exp() * wi + ln_kjj.exp() * wj) / (wi + wj) }
        }
    };
    Some(kij / r.powf(BOND_POWER))
}

/// Angle force constant of eq. (5) as corrected in the erratum, with `D`
/// of eq. (6): `143.9 Z_i C_j Z_k / ((r_ij + r_jk) sqrt(θ) exp(2D))`, for
/// the angle `i-j-k` with equilibrium bond lengths `r_ij`, `r_jk` (Å) and
/// equilibrium angle `theta_deg` (θ enters the formula in radians).
///
/// The original paper prints `θ⁻²`; the erratum corrects the exponent to
/// `-1/2`. With it the formula reproduces the angle force constants of
/// GAFF 1.81 to within 0.1 % (median ratio 1.0000 over 5299 angles). The
/// GAFF 2.x tables were refitted and are on average 13-25 % stiffer than
/// this rule.
pub fn angle_force_constant(zi: u8, zj: u8, zk: u8, r_ij: f64, r_jk: f64, theta_deg: f64) -> Option<f64> {
    let (zi, zk) = (TABLE4_Z[slot(zi)?], TABLE4_Z[slot(zk)?]);
    let cj = table4_c(zj)?;
    let theta = theta_deg.to_radians();
    let sum = r_ij + r_jk;
    if sum <= 0.0 || theta <= 0.0 {
        return None;
    }
    let d = (r_ij - r_jk) * (r_ij - r_jk) / (sum * sum);
    Some(143.9 * zi * cj * zk / (sum * theta.sqrt()) * (-2.0 * d).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bond_rule_reproduces_table_values() {
        // c3-c3 at 1.535 Å: the paper-era value is 303.1
        let k = bond_force_constant(6, 6, 1.535).unwrap();
        assert!((k - 303.1).abs() < 1.0, "{k}");
        // H-H (Table 2: fitted to the H2 stretch)
        let k = bond_force_constant(1, 1, 0.738).unwrap();
        assert!((k - 4.661_f64.exp() / 0.738_f64.powf(4.5)).abs() < 1e-9);
        // pairs missing from Table 3 use eq. (4)
        let k = bond_force_constant(9, 35, 1.76).unwrap();
        let (kf, kbr) = (7.358_f64.exp(), 9.012_f64.exp());
        assert!(k * 1.76_f64.powf(4.5) > kf && k * 1.76_f64.powf(4.5) < kbr);
        assert!(bond_force_constant(6, 14, 1.9).is_none());
    }

    #[test]
    fn angle_rule() {
        // tabulated GAFF 1.81 values: c3-c3-c3 62.9 (111.51°, 1.5375 Å),
        // hc-c3-hc 39.4 (107.58°, 1.0969 Å), c3-c3-hc 46.3 (109.80°)
        let k = angle_force_constant(6, 6, 6, 1.5375, 1.5375, 111.51).unwrap();
        assert!((k - 62.9).abs() < 0.05, "{k}");
        let k = angle_force_constant(1, 6, 1, 1.0969, 1.0969, 107.58).unwrap();
        assert!((k - 39.4).abs() < 0.05, "{k}");
        let k = angle_force_constant(6, 6, 1, 1.5375, 1.0969, 109.80).unwrap();
        assert!((k - 46.3).abs() < 0.05, "{k}");
        // unequal bonds are softer (eq. 6)
        assert!(angle_force_constant(1, 6, 17, 1.09, 1.80, 108.0).unwrap() < angle_force_constant(1, 6, 17, 1.445, 1.445, 108.0).unwrap());
        assert!(angle_force_constant(6, 9, 6, 1.4, 1.4, 109.0).is_none());
    }
}
