//! MDL Molfile (V2000 / V3000) and SD file reading/writing.

use crate::core::{Atom, BondDir, BondOrder, Molecule, atomic_number};
use crate::error::{Error, Result};
use crate::perception;
use std::fmt::Write as _;

fn field(line: &str, start: usize, len: usize) -> &str {
    let b = line.as_bytes();
    if start >= b.len() {
        return "";
    }
    let end = (start + len).min(b.len());
    std::str::from_utf8(&b[start..end]).unwrap_or("").trim()
}

fn parse_f(s: &str, line: usize) -> Result<f64> {
    s.trim().parse::<f64>().map_err(|_| Error::parse(line, format!("bad number '{s}'")))
}
fn parse_u(s: &str, line: usize) -> Result<usize> {
    s.trim().parse::<usize>().map_err(|_| Error::parse(line, format!("bad integer '{s}'")))
}

fn bond_from_code(code: i32, line: usize) -> Result<BondOrder> {
    Ok(match code {
        1 => BondOrder::Single,
        2 => BondOrder::Double,
        3 => BondOrder::Triple,
        4 => BondOrder::Aromatic,
        9 => BondOrder::Dative,
        0 | 10 => BondOrder::Zero,
        5..=8 => BondOrder::Single, // query bond types: treat as single
        _ => return Err(Error::parse(line, format!("unsupported bond type {code}"))),
    })
}

/// Parses one molfile block (without the SDF data part). `base_line` is used
/// for error messages.
fn parse_molblock(lines: &[&str], base_line: usize) -> Result<Molecule> {
    if lines.len() < 4 {
        return Err(Error::parse(base_line + 1, "molfile too short"));
    }
    let mut mol = Molecule::new();
    mol.name = lines[0].trim().to_string();
    let counts = lines[3];
    if counts.contains("V3000") {
        parse_v3000(&mut mol, lines, base_line)?;
    } else {
        parse_v2000(&mut mol, lines, base_line)?;
    }
    perception::sanitize(&mut mol)?;
    Ok(mol)
}

fn parse_v2000(mol: &mut Molecule, lines: &[&str], base: usize) -> Result<()> {
    let counts = lines[3];
    let na = parse_u(field(counts, 0, 3), base + 4)?;
    let nb = parse_u(field(counts, 3, 3), base + 4)?;
    if lines.len() < 4 + na + nb {
        return Err(Error::parse(base + 4, "truncated atom/bond block"));
    }
    let mut block_charges = Vec::with_capacity(na);
    for k in 0..na {
        let ln = 4 + k;
        let l = lines[ln];
        let x = parse_f(field(l, 0, 10), base + ln + 1)?;
        let y = parse_f(field(l, 10, 10), base + ln + 1)?;
        let z = parse_f(field(l, 20, 10), base + ln + 1)?;
        let sym = field(l, 31, 3);
        let anum = if sym == "D" || sym == "T" {
            1
        } else {
            atomic_number(sym).ok_or_else(|| Error::UnknownElement(sym.to_string()))?
        };
        let mut a = Atom::new(anum);
        let chg_code: i32 = field(l, 36, 3).parse().unwrap_or(0);
        let chg = match chg_code {
            1 => 3,
            2 => 2,
            3 => 1,
            5 => -1,
            6 => -2,
            7 => -3,
            _ => 0,
        };
        if chg_code == 4 {
            a.num_radical_electrons = 1;
        }
        block_charges.push(chg);
        let valence: i32 = field(l, 48, 3).parse().unwrap_or(0);
        if valence == 15 {
            a.no_implicit = true;
        }
        let hcount: i32 = field(l, 42, 3).parse().unwrap_or(0);
        if hcount == 1 {
            // H0 query: no implicit hydrogens
            a.no_implicit = true;
        }
        mol.add_atom(a, [x, y, z]);
    }
    for k in 0..nb {
        let ln = 4 + na + k;
        let l = lines[ln];
        let a1 = parse_u(field(l, 0, 3), base + ln + 1)?;
        let a2 = parse_u(field(l, 3, 3), base + ln + 1)?;
        let t: i32 = field(l, 6, 3).parse().map_err(|_| Error::parse(base + ln + 1, "bad bond type"))?;
        if a1 == 0 || a2 == 0 || a1 > na || a2 > na {
            return Err(Error::parse(base + ln + 1, "bond atom index out of range"));
        }
        let b = mol.add_bond(a1 - 1, a2 - 1, bond_from_code(t, base + ln + 1)?);
        let stereo: i32 = field(l, 9, 3).parse().unwrap_or(0);
        mol.bond_mut(b).dir = match stereo {
            1 => BondDir::Up,
            6 => BondDir::Down,
            // 4: wavy single bond; 3: crossed (either) double bond
            3 | 4 => BondDir::Either,
            _ => BondDir::None,
        };
    }
    // properties block
    let mut has_chg = false;
    for (off, l) in lines.iter().enumerate().skip(4 + na + nb) {
        if l.starts_with("M  END") {
            break;
        }
        let tag = l.get(0..6).unwrap_or("");
        let items: Vec<&str> = l.get(6..).unwrap_or("").split_whitespace().collect();
        let pairs = |items: &[&str]| -> Vec<(usize, i32)> {
            let mut v = Vec::new();
            let n: usize = items.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            for k in 0..n {
                let a: usize = items.get(1 + 2 * k).and_then(|s| s.parse().ok()).unwrap_or(0);
                let b: i32 = items.get(2 + 2 * k).and_then(|s| s.parse().ok()).unwrap_or(0);
                if a > 0 {
                    v.push((a - 1, b));
                }
            }
            v
        };
        match tag {
            "M  CHG" => {
                if !has_chg {
                    has_chg = true;
                    for a in mol.atoms_mut() {
                        a.formal_charge = 0;
                    }
                }
                for (a, c) in pairs(&items) {
                    if a < mol.num_atoms() {
                        mol.atom_mut(a).formal_charge = c as i8;
                    }
                }
            }
            "M  RAD" => {
                for (a, r) in pairs(&items) {
                    if a < mol.num_atoms() {
                        mol.atom_mut(a).num_radical_electrons = match r {
                            2 => 1,
                            1 | 3 => 2,
                            _ => 0,
                        };
                    }
                }
            }
            "M  ISO" => {
                for (a, m) in pairs(&items) {
                    if a < mol.num_atoms() {
                        mol.atom_mut(a).isotope = m as u16;
                    }
                }
            }
            _ => {
                let _ = off;
            }
        }
    }
    if !has_chg {
        for (i, c) in block_charges.into_iter().enumerate() {
            mol.atom_mut(i).formal_charge = c;
        }
    }
    Ok(())
}

fn parse_v3000(mol: &mut Molecule, lines: &[&str], base: usize) -> Result<()> {
    // join continuation lines
    let mut v30: Vec<(usize, String)> = Vec::new();
    let mut pending: Option<(usize, String)> = None;
    for (k, l) in lines.iter().enumerate().skip(4) {
        if l.starts_with("M  END") {
            break;
        }
        let Some(rest) = l.strip_prefix("M  V30 ") else { continue };
        let (ln, mut s) = pending.take().unwrap_or((base + k + 1, String::new()));
        if let Some(r) = rest.strip_suffix('-') {
            s.push_str(r);
            pending = Some((ln, s));
        } else {
            s.push_str(rest);
            v30.push((ln, s));
        }
    }
    let mut section = "";
    let mut idx_map = std::collections::HashMap::new();
    for (ln, l) in &v30 {
        let t = l.trim();
        if let Some(s) = t.strip_prefix("BEGIN ") {
            section = match s {
                "ATOM" => "ATOM",
                "BOND" => "BOND",
                _ => "OTHER",
            };
            continue;
        }
        if t.starts_with("END ") {
            section = "";
            continue;
        }
        let f: Vec<&str> = t.split_whitespace().collect();
        match section {
            "ATOM" => {
                if f.len() < 6 {
                    return Err(Error::parse(*ln, "bad V3000 atom line"));
                }
                let sym = f[1];
                let anum = atomic_number(sym).ok_or_else(|| Error::UnknownElement(sym.to_string()))?;
                let mut a = Atom::new(anum);
                for kv in &f[6..] {
                    if let Some((k, v)) = kv.split_once('=') {
                        match k {
                            "CHG" => a.formal_charge = v.parse().unwrap_or(0),
                            "RAD" => {
                                a.num_radical_electrons = match v {
                                    "2" => 1,
                                    "1" | "3" => 2,
                                    _ => 0,
                                }
                            }
                            "MASS" => a.isotope = v.parse().unwrap_or(0),
                            "HCOUNT" => {
                                if v == "-1" {
                                    a.no_implicit = true
                                }
                            }
                            "VAL" => {
                                if v == "-1" {
                                    a.no_implicit = true
                                }
                            }
                            _ => {}
                        }
                    }
                }
                let i = mol.add_atom(a, [parse_f(f[2], *ln)?, parse_f(f[3], *ln)?, parse_f(f[4], *ln)?]);
                idx_map.insert(f[0].to_string(), i);
            }
            "BOND" => {
                if f.len() < 4 {
                    return Err(Error::parse(*ln, "bad V3000 bond line"));
                }
                let t: i32 = f[1].parse().map_err(|_| Error::parse(*ln, "bad bond type"))?;
                let a1 = *idx_map.get(f[2]).ok_or_else(|| Error::parse(*ln, "unknown atom index"))?;
                let a2 = *idx_map.get(f[3]).ok_or_else(|| Error::parse(*ln, "unknown atom index"))?;
                let b = mol.add_bond(a1, a2, bond_from_code(t, *ln)?);
                let cfg = f[4..].iter().find_map(|kv| kv.strip_prefix("CFG="));
                mol.bond_mut(b).dir = match cfg {
                    Some("1") => BondDir::Up,
                    Some("3") => BondDir::Down,
                    Some("2") => BondDir::Either,
                    _ => BondDir::None,
                };
            }
            _ => {}
        }
    }
    Ok(())
}

/// Reads a single molfile.
pub fn read_molfile(text: &str) -> Result<Molecule> {
    let lines: Vec<&str> = text.lines().collect();
    parse_molblock(&lines, 0)
}

/// Reads all molecules from an SD file. Data fields go into `Molecule::props`.
pub fn read_sdf(text: &str) -> Result<Vec<Molecule>> {
    read_sdf_lenient(text).into_iter().collect()
}

/// Like [`read_sdf`] but returns a result per record so that one bad record
/// does not abort the whole file.
pub fn read_sdf_lenient(text: &str) -> Vec<Result<Molecule>> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        // find end of record
        let mut end = start;
        while end < lines.len() && lines[end].trim_end() != "$$$$" {
            end += 1;
        }
        let rec = &lines[start..end];
        if rec.iter().any(|l| !l.trim().is_empty()) {
            let mend = rec.iter().position(|l| l.starts_with("M  END")).unwrap_or(rec.len());
            let mut res = parse_molblock(&rec[..(mend + 1).min(rec.len())], start);
            if let Ok(mol) = res.as_mut() {
                parse_data_fields(mol, &rec[(mend + 1).min(rec.len())..]);
            }
            out.push(res);
        }
        start = end + 1;
    }
    out
}

fn parse_data_fields(mol: &mut Molecule, lines: &[&str]) {
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        if l.starts_with('>') {
            if let (Some(a), Some(b)) = (l.find('<'), l.rfind('>')) {
                if b > a {
                    let name = l[a + 1..b].to_string();
                    let mut val = Vec::new();
                    i += 1;
                    while i < lines.len() && !lines[i].trim().is_empty() {
                        val.push(lines[i]);
                        i += 1;
                    }
                    mol.props.insert(name, val.join("\n"));
                    continue;
                }
            }
        }
        i += 1;
    }
}

/// Writes a V2000 molblock (aromatic bonds are kekulized when possible).
pub fn write_molblock(mol: &Molecule) -> String {
    write_molblock_coords(mol, &mol.coords)
}

/// Writes a V2000 molblock with explicit coordinates.
pub fn write_molblock_coords(mol: &Molecule, coords: &[[f64; 3]]) -> String {
    let km = perception::kekulized_copy(mol).unwrap_or_else(|_| mol.clone());
    let mut s = String::new();
    let _ = writeln!(s, "{}", mol.name);
    let _ = writeln!(s, "  molecule_optimizer_rs   3D");
    let _ = writeln!(s);
    let _ = writeln!(s, "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000", km.num_atoms(), km.num_bonds());
    for (a, p) in km.atoms().iter().zip(coords) {
        let _ = writeln!(
            s,
            "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0",
            p[0], p[1], p[2], a.symbol()
        );
    }
    for b in km.bonds() {
        let t = match b.order {
            BondOrder::Single => 1,
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
            BondOrder::Aromatic => 4,
            BondOrder::Dative => 9,
            BondOrder::Zero => 0,
        };
        let _ = writeln!(s, "{:>3}{:>3}{:>3}  0", b.begin + 1, b.end + 1, t);
    }
    let charged: Vec<(usize, i8)> =
        km.atoms().iter().enumerate().filter(|(_, a)| a.formal_charge != 0).map(|(i, a)| (i, a.formal_charge)).collect();
    for chunk in charged.chunks(8) {
        let _ = write!(s, "M  CHG{:>3}", chunk.len());
        for (i, c) in chunk {
            let _ = write!(s, " {:>3} {:>3}", i + 1, c);
        }
        let _ = writeln!(s);
    }
    let _ = writeln!(s, "M  END");
    s
}

/// Writes an SDF record (molblock + data fields + `$$$$`).
pub fn write_sdf_record(mol: &Molecule, coords: Option<&[[f64; 3]]>, extra: &[(&str, String)]) -> String {
    let mut s = write_molblock_coords(mol, coords.unwrap_or(&mol.coords));
    for (k, v) in mol.props.iter() {
        let _ = writeln!(s, "> <{}>\n{}\n", k, v);
    }
    for (k, v) in extra {
        let _ = writeln!(s, "> <{}>\n{}\n", k, v);
    }
    s.push_str("$$$$\n");
    s
}
