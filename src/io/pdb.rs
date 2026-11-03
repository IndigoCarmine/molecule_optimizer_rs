//! PDB format reading/writing.

use crate::core::{Atom, BondOrder, Molecule, ResidueInfo, atomic_number};
use crate::error::{Error, Result};
use crate::perception;
use std::collections::HashMap;
use std::fmt::Write as _;

fn col(line: &str, a: usize, b: usize) -> &str {
    let bytes = line.as_bytes();
    if a >= bytes.len() {
        return "";
    }
    std::str::from_utf8(&bytes[a..b.min(bytes.len())]).unwrap_or("")
}

fn element_from(line: &str, name: &str, hetatm: bool) -> Option<u8> {
    let e = col(line, 76, 78).trim();
    if !e.is_empty() {
        if let Some(z) = atomic_number(e) {
            return Some(z);
        }
    }
    // derive from atom name: columns 12-13 hold the element right-justified
    let raw = col(line, 12, 16);
    let letters: String = raw.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if letters.is_empty() {
        return None;
    }
    if hetatm || raw.starts_with(|c: char| c.is_ascii_alphabetic()) {
        if letters.len() >= 2 {
            if let Some(z) = atomic_number(&letters[..2]) {
                // avoid e.g. "CA" (alpha carbon) in ATOM records
                if hetatm {
                    return Some(z);
                }
            }
        }
    }
    let _ = name;
    atomic_number(&letters[..1])
}

/// Reads all models from a PDB file. Connectivity comes from CONECT records
/// plus distance-based perception; bond orders are assigned heuristically.
pub fn read_pdb_models(text: &str) -> Result<Vec<Molecule>> {
    let mut models = Vec::new();
    let mut cur = Molecule::new();
    let mut serial_map: HashMap<i32, usize> = HashMap::new();
    let mut conects: Vec<(i32, i32)> = Vec::new();
    let mut name = String::new();
    let finish = |mol: &mut Molecule, smap: &HashMap<i32, usize>, conects: &[(i32, i32)]| -> Result<Molecule> {
        let mut m = std::mem::take(mol);
        for &(a, b) in conects {
            if let (Some(&i), Some(&j)) = (smap.get(&a), smap.get(&b)) {
                if i != j {
                    m.add_bond(i, j, BondOrder::Single);
                }
            }
        }
        perception::perceive_from_coordinates(&mut m)?;
        Ok(m)
    };
    for (ln, line) in text.lines().enumerate() {
        let rec = col(line, 0, 6);
        match rec.trim_end() {
            "HEADER" | "COMPND" => {
                if name.is_empty() {
                    name = col(line, 10, 80).trim().to_string();
                }
            }
            "ATOM" | "HETATM" => {
                let het = rec.starts_with("HETATM");
                let alt = col(line, 16, 17).chars().next().unwrap_or(' ');
                if alt != ' ' && alt != 'A' {
                    continue;
                }
                let atom_name = col(line, 12, 16).trim().to_string();
                let z = element_from(line, &atom_name, het)
                    .ok_or_else(|| Error::parse(ln + 1, format!("can't determine element for '{atom_name}'")))?;
                let p = |a, b| -> Result<f64> {
                    col(line, a, b).trim().parse::<f64>().map_err(|_| Error::parse(ln + 1, "bad coordinate"))
                };
                let pos = [p(30, 38)?, p(38, 46)?, p(46, 54)?];
                let mut atom = Atom::new(z);
                let chg = col(line, 78, 80).trim();
                if chg.len() == 2 {
                    let (d, s) = chg.split_at(1);
                    if let Ok(v) = d.parse::<i8>() {
                        atom.formal_charge = if s == "-" { -v } else { v };
                    }
                }
                let serial: i32 = col(line, 6, 11).trim().parse().unwrap_or(cur.num_atoms() as i32 + 1);
                atom.residue = Some(ResidueInfo {
                    atom_name,
                    residue_name: col(line, 17, 20).trim().to_string(),
                    residue_number: col(line, 22, 26).trim().parse().unwrap_or(0),
                    chain_id: col(line, 21, 22).chars().next().unwrap_or(' '),
                    insertion_code: col(line, 26, 27).chars().next().unwrap_or(' '),
                    serial,
                    is_hetatm: het,
                    occupancy: col(line, 54, 60).trim().parse().unwrap_or(1.0),
                    temp_factor: col(line, 60, 66).trim().parse().unwrap_or(0.0),
                    alt_loc: alt,
                });
                let idx = cur.add_atom(atom, pos);
                serial_map.insert(serial, idx);
            }
            "CONECT" => {
                let a: i32 = match col(line, 6, 11).trim().parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                for k in 0..4 {
                    let s = 11 + 5 * k;
                    if let Ok(b) = col(line, s, s + 5).trim().parse::<i32>() {
                        conects.push((a, b));
                    }
                }
            }
            "ENDMDL" => {
                if cur.num_atoms() > 0 {
                    let mut m = finish(&mut cur, &serial_map, &conects)?;
                    m.name = name.clone();
                    models.push(m);
                    serial_map.clear();
                }
            }
            _ => {}
        }
    }
    if cur.num_atoms() > 0 {
        let mut m = finish(&mut cur, &serial_map, &conects)?;
        m.name = name;
        models.push(m);
    }
    if models.is_empty() {
        return Err(Error::parse(1, "no atoms in PDB"));
    }
    Ok(models)
}

/// Reads the first model of a PDB file.
pub fn read_pdb(text: &str) -> Result<Molecule> {
    Ok(read_pdb_models(text)?.remove(0))
}

/// Writes a PDB file (HETATM/ATOM + CONECT).
pub fn write_pdb(mol: &Molecule) -> String {
    write_pdb_coords(mol, &mol.coords)
}

pub fn write_pdb_coords(mol: &Molecule, coords: &[[f64; 3]]) -> String {
    let mut s = String::new();
    if !mol.name.is_empty() {
        let _ = writeln!(s, "COMPND    {}", mol.name);
    }
    for (i, (a, p)) in mol.atoms().iter().zip(coords).enumerate() {
        let serial = i + 1;
        let (rec, name, resn, resi, chain) = match &a.residue {
            Some(r) => (
                if r.is_hetatm { "HETATM" } else { "ATOM  " },
                r.atom_name.clone(),
                r.residue_name.clone(),
                r.residue_number,
                r.chain_id,
            ),
            None => ("HETATM", format!("{}{}", a.symbol(), serial), "UNL".to_string(), 1, ' '),
        };
        let name_fmt = if name.len() < 4 && a.symbol().len() == 1 { format!(" {:<3}", name) } else { format!("{:<4}", name) };
        let chg = match a.formal_charge {
            0 => "  ".to_string(),
            c if c > 0 => format!("{}+", c),
            c => format!("{}-", -c),
        };
        let _ = writeln!(
            s,
            "{}{:>5} {:4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}{}",
            rec,
            serial % 100000,
            &name_fmt[..name_fmt.len().min(4)],
            &resn[..resn.len().min(3)],
            chain,
            resi,
            p[0],
            p[1],
            p[2],
            1.0,
            0.0,
            a.symbol().to_uppercase(),
            chg
        );
    }
    for i in 0..mol.num_atoms() {
        let nb: Vec<usize> = mol.neighbors(i).collect();
        for chunk in nb.chunks(4) {
            let _ = write!(s, "CONECT{:>5}", i + 1);
            for &j in chunk {
                let _ = write!(s, "{:>5}", j + 1);
            }
            let _ = writeln!(s);
        }
    }
    let _ = writeln!(s, "END");
    s
}
