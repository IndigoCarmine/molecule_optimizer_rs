//! XYZ format (single or multi-frame).

use crate::core::{Atom, Molecule, atomic_number};
use crate::error::{Error, Result};
use crate::perception;
use std::fmt::Write as _;

/// Parses all frames of an XYZ file. Connectivity and bond orders are
/// perceived from distances (see [`perception::perceive_from_coordinates`]).
pub fn read_xyz_str(text: &str) -> Result<Vec<Molecule>> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }
        let n: usize = lines[i].trim().parse().map_err(|_| Error::parse(i + 1, "expected atom count"))?;
        if i + 1 + n > lines.len() {
            return Err(Error::parse(i + 1, "truncated XYZ frame"));
        }
        let mut mol = Molecule::new();
        mol.name = lines.get(i + 1).map(|s| s.trim().to_string()).unwrap_or_default();
        for k in 0..n {
            let ln = i + 2 + k;
            let l = lines.get(ln).ok_or_else(|| Error::parse(ln + 1, "missing atom line"))?;
            let f: Vec<&str> = l.split_whitespace().collect();
            if f.len() < 4 {
                return Err(Error::parse(ln + 1, "atom line needs symbol x y z"));
            }
            let anum = match f[0].parse::<u8>() {
                Ok(z) => z,
                Err(_) => atomic_number(f[0]).ok_or_else(|| Error::UnknownElement(f[0].to_string()))?,
            };
            let p = |s: &str| s.parse::<f64>().map_err(|_| Error::parse(ln + 1, format!("bad number '{s}'")));
            let mut a = Atom::new(anum);
            a.no_implicit = true;
            mol.add_atom(a, [p(f[1])?, p(f[2])?, p(f[3])?]);
        }
        perception::perceive_from_coordinates(&mut mol)?;
        out.push(mol);
        i += 2 + n;
    }
    Ok(out)
}

/// Reads a single molecule (the first frame).
pub fn read_xyz(text: &str) -> Result<Molecule> {
    read_xyz_str(text)?.into_iter().next().ok_or_else(|| Error::parse(1, "empty XYZ"))
}

/// Writes one XYZ frame. `comment` replaces the molecule name if given.
pub fn write_xyz(mol: &Molecule, comment: Option<&str>) -> String {
    write_xyz_coords(mol, &mol.coords, comment)
}

/// Writes one XYZ frame with explicit coordinates (e.g. from a snapshot).
pub fn write_xyz_coords(mol: &Molecule, coords: &[[f64; 3]], comment: Option<&str>) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "{}", mol.num_atoms());
    let _ = writeln!(s, "{}", comment.unwrap_or(&mol.name));
    for (a, p) in mol.atoms().iter().zip(coords) {
        let _ = writeln!(s, "{:<3} {:>14.8} {:>14.8} {:>14.8}", a.symbol(), p[0], p[1], p[2]);
    }
    s
}
