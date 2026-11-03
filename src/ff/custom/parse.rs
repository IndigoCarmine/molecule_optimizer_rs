//! Lexer, parser and printer of the expression language (see the module
//! documentation of [`crate::ff::custom`] for the grammar).

use std::fmt;

/// A parse or resolution error with the byte offset into the source.
#[derive(Debug, Clone, PartialEq)]
pub struct ExprError {
    /// Byte offset of the offending token in `source`.
    pub pos: usize,
    pub msg: String,
    pub source: String,
}

impl ExprError {
    pub(crate) fn new(pos: usize, msg: impl Into<String>, source: &str) -> Self {
        ExprError { pos, msg: msg.into(), source: source.to_string() }
    }
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show the line that contains the error with a caret below it.
        let pos = self.pos.min(self.source.len());
        let start = self.source[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let end = self.source[pos..].find('\n').map(|i| pos + i).unwrap_or(self.source.len());
        let line = self.source[start..end].trim_end_matches('\r');
        let col = self.source[start..pos].chars().count();
        write!(f, "{} at position {}\n  {}\n  {}^", self.msg, self.pos, line, " ".repeat(col))
    }
}

impl std::error::Error for ExprError {}

impl From<ExprError> for crate::error::Error {
    fn from(e: ExprError) -> Self {
        crate::error::Error::InvalidArgument(format!("expression error: {e}"))
    }
}

/// Binary operators of the surface syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl BinOp {
    fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Pow => "^",
        }
    }
    fn prec(self) -> u8 {
        match self {
            BinOp::Add | BinOp::Sub => 1,
            BinOp::Mul | BinOp::Div => 2,
            BinOp::Pow => 4,
        }
    }
}

/// Expression syntax tree with source positions.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    /// Byte offset of the first character of this expression.
    pub pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

/// `name = expr` (definition) or a bare `expr`.
#[derive(Debug, Clone, PartialEq)]
pub struct Statement {
    /// Defined name and its position.
    pub name: Option<(String, usize)>,
    pub expr: Expr,
}

/// A parsed source: `;`-separated statements. The first statement is the
/// result; the others are definitions.
#[derive(Debug, Clone, PartialEq)]
pub struct Script {
    pub statements: Vec<Statement>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Sym(char),
    Pow,
    End,
}

fn describe(t: &Tok) -> String {
    match t {
        Tok::Num(v) => format!("number {v}"),
        Tok::Ident(s) => format!("identifier '{s}'"),
        Tok::Sym(c) => format!("'{c}'"),
        Tok::Pow => "'^'".to_string(),
        Tok::End => "end of input".to_string(),
    }
}

fn lex(src: &str) -> Result<Vec<(Tok, usize)>, ExprError> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < b.len() && b[i] == b'.' {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    while j < b.len() && b[j].is_ascii_digit() {
                        j += 1;
                    }
                    i = j;
                } else {
                    return Err(ExprError::new(i, "malformed exponent in number", src));
                }
            }
            if i < b.len() && (b[i].is_ascii_alphabetic() || b[i] == b'_') {
                return Err(ExprError::new(i, "unexpected character after number", src));
            }
            let v: f64 = src[start..i].parse().map_err(|_| ExprError::new(start, "invalid number", src))?;
            out.push((Tok::Num(v), start));
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push((Tok::Ident(src[start..i].to_string()), start));
            continue;
        }
        match c {
            b'*' if i + 1 < b.len() && b[i + 1] == b'*' => {
                out.push((Tok::Pow, start));
                i += 2;
            }
            b'^' => {
                out.push((Tok::Pow, start));
                i += 1;
            }
            b'+' | b'-' | b'*' | b'/' | b'(' | b')' | b',' | b';' | b'=' => {
                out.push((Tok::Sym(c as char), start));
                i += 1;
            }
            _ => {
                let ch = src[i..].chars().next().unwrap_or('?');
                return Err(ExprError::new(i, format!("unexpected character '{ch}'"), src));
            }
        }
    }
    out.push((Tok::End, src.len()));
    Ok(out)
}

struct Parser<'a> {
    toks: Vec<(Tok, usize)>,
    i: usize,
    src: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> &Tok {
        &self.toks[self.i].0
    }
    fn pos(&self) -> usize {
        self.toks[self.i].1
    }
    fn bump(&mut self) -> (Tok, usize) {
        let t = self.toks[self.i].clone();
        if self.i + 1 < self.toks.len() {
            self.i += 1;
        }
        t
    }
    fn err<T>(&self, msg: impl Into<String>) -> Result<T, ExprError> {
        Err(ExprError::new(self.pos(), msg, self.src))
    }
    fn expect(&mut self, c: char) -> Result<(), ExprError> {
        if *self.peek() == Tok::Sym(c) {
            self.bump();
            Ok(())
        } else {
            self.err(format!("expected '{c}', found {}", describe(self.peek())))
        }
    }

    fn script(&mut self) -> Result<Vec<Statement>, ExprError> {
        let mut stmts = Vec::new();
        loop {
            while *self.peek() == Tok::Sym(';') {
                self.bump();
            }
            if *self.peek() == Tok::End {
                break;
            }
            stmts.push(self.statement()?);
            match self.peek() {
                Tok::Sym(';') | Tok::End => {}
                t => return self.err(format!("expected ';' or end of input, found {}", describe(t))),
            }
        }
        if stmts.is_empty() {
            return self.err("empty expression");
        }
        Ok(stmts)
    }

    fn statement(&mut self) -> Result<Statement, ExprError> {
        if let Tok::Ident(name) = self.peek().clone()
            && self.toks.get(self.i + 1).map(|t| &t.0) == Some(&Tok::Sym('='))
        {
            let p = self.pos();
            self.bump();
            self.bump();
            let expr = self.expr()?;
            return Ok(Statement { name: Some((name, p)), expr });
        }
        let expr = self.expr()?;
        if *self.peek() == Tok::Sym('=') {
            return self.err("left side of '=' must be a single name");
        }
        Ok(Statement { name: None, expr })
    }

    fn expr(&mut self) -> Result<Expr, ExprError> {
        let mut lhs = self.term()?;
        loop {
            let op = match self.peek() {
                Tok::Sym('+') => BinOp::Add,
                Tok::Sym('-') => BinOp::Sub,
                _ => return Ok(lhs),
            };
            self.bump();
            let rhs = self.term()?;
            let pos = lhs.pos;
            lhs = Expr { kind: ExprKind::Bin(op, Box::new(lhs), Box::new(rhs)), pos };
        }
    }

    fn term(&mut self) -> Result<Expr, ExprError> {
        let mut lhs = self.unary()?;
        loop {
            let op = match self.peek() {
                Tok::Sym('*') => BinOp::Mul,
                Tok::Sym('/') => BinOp::Div,
                _ => return Ok(lhs),
            };
            self.bump();
            let rhs = self.unary()?;
            let pos = lhs.pos;
            lhs = Expr { kind: ExprKind::Bin(op, Box::new(lhs), Box::new(rhs)), pos };
        }
    }

    fn unary(&mut self) -> Result<Expr, ExprError> {
        match self.peek() {
            Tok::Sym('-') => {
                let p = self.pos();
                self.bump();
                let e = self.unary()?;
                Ok(Expr { kind: ExprKind::Neg(Box::new(e)), pos: p })
            }
            Tok::Sym('+') => {
                self.bump();
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Expr, ExprError> {
        let base = self.primary()?;
        if *self.peek() == Tok::Pow {
            self.bump();
            // Right associative; the exponent may carry a sign (`x^-2`).
            let exp = self.unary()?;
            let pos = base.pos;
            return Ok(Expr { kind: ExprKind::Bin(BinOp::Pow, Box::new(base), Box::new(exp)), pos });
        }
        Ok(base)
    }

    fn primary(&mut self) -> Result<Expr, ExprError> {
        let (t, p) = self.bump();
        match t {
            Tok::Num(v) => Ok(Expr { kind: ExprKind::Num(v), pos: p }),
            Tok::Ident(name) => {
                if *self.peek() == Tok::Sym('(') {
                    self.bump();
                    let mut args = Vec::new();
                    if *self.peek() != Tok::Sym(')') {
                        loop {
                            args.push(self.expr()?);
                            if *self.peek() == Tok::Sym(',') {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(')')?;
                    Ok(Expr { kind: ExprKind::Call(name, args), pos: p })
                } else {
                    Ok(Expr { kind: ExprKind::Var(name), pos: p })
                }
            }
            Tok::Sym('(') => {
                let e = self.expr()?;
                self.expect(')')?;
                Ok(e)
            }
            t => {
                self.i -= usize::from(self.i > 0 && t != Tok::End);
                self.err(format!("expected a number, name or '(', found {}", describe(&t)))
            }
        }
    }
}

/// Parses a `;`-separated script.
pub fn parse(src: &str) -> Result<Script, ExprError> {
    let toks = lex(src)?;
    let mut p = Parser { toks, i: 0, src };
    let statements = p.script()?;
    Ok(Script { statements, source: src.to_string() })
}

/// Parses a single expression (no `;` or definitions).
pub fn parse_expr(src: &str) -> Result<Expr, ExprError> {
    let toks = lex(src)?;
    let mut p = Parser { toks, i: 0, src };
    let e = p.expr()?;
    if *p.peek() != Tok::End {
        return p.err(format!("expected end of input, found {}", describe(p.peek())));
    }
    Ok(e)
}

/// Parses a list of definitions `a = ...; b = ...` (every statement must be
/// a definition).
pub fn parse_definitions(src: &str) -> Result<Script, ExprError> {
    if src.trim().is_empty() {
        return Ok(Script { statements: Vec::new(), source: src.to_string() });
    }
    let s = parse(src)?;
    for st in &s.statements {
        if st.name.is_none() {
            return Err(ExprError::new(st.expr.pos, "expected a definition 'name = expression'", src));
        }
    }
    Ok(s)
}

pub(crate) fn format_number(v: f64) -> String {
    if v != 0.0 && (v.abs() >= 1e15 || v.abs() < 1e-5) { format!("{v:e}") } else { format!("{v}") }
}

fn prec(e: &Expr) -> u8 {
    match &e.kind {
        ExprKind::Num(_) | ExprKind::Var(_) | ExprKind::Call(..) => 5,
        ExprKind::Neg(_) => 3,
        ExprKind::Bin(op, ..) => op.prec(),
    }
}

fn write_child(out: &mut String, e: &Expr, paren: bool) {
    if paren {
        out.push('(');
        write_expr(out, e);
        out.push(')');
    } else {
        write_expr(out, e);
    }
}

fn write_expr(out: &mut String, e: &Expr) {
    match &e.kind {
        ExprKind::Num(v) => out.push_str(&format_number(*v)),
        ExprKind::Var(s) => out.push_str(s),
        ExprKind::Neg(a) => {
            out.push('-');
            write_child(out, a, prec(a) < 3);
        }
        ExprKind::Bin(op, a, b) => {
            let p = op.prec();
            if *op == BinOp::Pow {
                write_child(out, a, prec(a) <= p);
                out.push('^');
                write_child(out, b, prec(b) < p);
            } else {
                write_child(out, a, prec(a) < p);
                if p == 1 {
                    out.push(' ');
                    out.push_str(op.symbol());
                    out.push(' ');
                } else {
                    out.push_str(op.symbol());
                }
                write_child(out, b, prec(b) <= p);
            }
        }
        ExprKind::Call(name, args) => {
            out.push_str(name);
            out.push('(');
            for (k, a) in args.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                write_expr(out, a);
            }
            out.push(')');
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        write_expr(&mut s, self);
        f.write_str(&s)
    }
}

impl fmt::Display for Script {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (k, st) in self.statements.iter().enumerate() {
            if k > 0 {
                f.write_str("; ")?;
            }
            if let Some((n, _)) = &st.name {
                write!(f, "{n} = ")?;
            }
            write!(f, "{}", st.expr)?;
        }
        Ok(())
    }
}

/// Structural equality ignoring positions.
pub fn same_structure(a: &Expr, b: &Expr) -> bool {
    match (&a.kind, &b.kind) {
        (ExprKind::Num(x), ExprKind::Num(y)) => x == y,
        (ExprKind::Var(x), ExprKind::Var(y)) => x == y,
        (ExprKind::Neg(x), ExprKind::Neg(y)) => same_structure(x, y),
        (ExprKind::Bin(o1, a1, b1), ExprKind::Bin(o2, a2, b2)) => {
            o1 == o2 && same_structure(a1, a2) && same_structure(b1, b2)
        }
        (ExprKind::Call(n1, x), ExprKind::Call(n2, y)) => {
            n1 == n2 && x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_structure(p, q))
        }
        _ => false,
    }
}
