//! Content tokens retain PDF decimal precision (lopdf's object Real is f32).
use crate::{Error, Result};
#[derive(Clone, Debug)]
pub enum Token {
    Int(i64),
    Real(f64),
    Name(String),
    Str(Vec<u8>, bool),
    Array(Vec<Token>),
    Dict(Vec<(String, Token)>),
    Word(String),
    Inline(Vec<(String, Token)>, Vec<u8>),
}
impl Token {
    // on_line_dash uses Python str(), including dataclass repr for malformed
    // nonnumeric operands; this is separate from PDF token serialization.
    pub(crate) fn python_str(&self) -> String {
        match self {
            Self::Int(n) => n.to_string(),
            Self::Real(n) => crate::python_float_str(*n),
            Self::Name(n) => format!("PdfName(value={})", python_string_repr(n)),
            Self::Str(bytes, hex) => format!(
                "PdfString(raw={}, is_hex={})",
                python_bytes_repr(bytes),
                if *hex { "True" } else { "False" }
            ),
            Self::Word(w) => format!("PdfKeyword(value={})", python_string_repr(w)),
            Self::Array(v) => format!(
                "[{}]",
                v.iter().map(Self::python_str).collect::<Vec<_>>().join(", ")
            ),
            Self::Dict(v) => format!(
                "{{{}}}",
                v.iter()
                    .map(|(k, v)| format!("{}: {}", python_string_repr(k), v.python_str()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Inline(..) => self.pdf(),
        }
    }
    pub fn number(&self) -> Result<f64> {
        match self {
            Self::Int(n) => Ok(*n as f64),
            Self::Real(n) => Ok(*n),
            _ => Err(Error::new("ValueError", "Expected numeric operand")),
        }
    }
    pub fn name(&self) -> Result<&str> {
        if let Self::Name(n) = self {
            Ok(n)
        } else {
            Err(Error::new("ValueError", "Expected PdfName operand"))
        }
    }
    pub fn pdf(&self) -> String {
        match self {
            Self::Int(n) => n.to_string(),
            Self::Real(n) => format!("{n:.6}"),
            Self::Name(n) => format!("/{n}"),
            Self::Str(s, true) => format!(
                "<{}>",
                s.iter().map(|x| format!("{x:02X}")).collect::<String>()
            ),
            Self::Str(s, false) => {
                let mut out = "(".to_owned();
                for &b in s {
                    match b {
                        8 => out.push_str("\\b"),
                        9 => out.push_str("\\t"),
                        10 => out.push_str("\\n"),
                        12 => out.push_str("\\f"),
                        13 => out.push_str("\\r"),
                        b'(' | b')' | b'\\' => {
                            out.push('\\');
                            out.push(b as char)
                        }
                        _ => out.push(b as char),
                    }
                }
                out.push(')');
                out
            }
            Self::Array(v) => format!(
                "[{}]",
                v.iter().map(Self::pdf).collect::<Vec<_>>().join(" ")
            ),
            Self::Dict(v) => format!(
                "<< {} >>",
                v.iter()
                    .map(|(k, v)| format!("/{k} {}", v.pdf()))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Self::Word(s) => s.clone(),
            Self::Inline(attrs, raw) => format!(
                "RuntimePdfStream(attrs={}, rawdata={}, objid=None, genno=None)",
                Self::Dict(attrs.clone()).inline_repr(),
                python_bytes_repr(raw),
            ),
        }
    }
    // Dataclass repr used by the reference's base-operation serializer for
    // RuntimePdfStream; this is deliberately not PDF stream syntax.
    fn inline_repr(&self) -> String {
        match self {
            Self::Int(n) => n.to_string(),
            Self::Real(n) => crate::python_float_str(*n),
            Self::Name(n) => format!("ActiveLiteral(name={})", python_string_repr(n)),
            Self::Str(bytes, _) => python_bytes_repr(bytes),
            Self::Array(a) => format!(
                "[{}]",
                a.iter()
                    .map(Self::inline_repr)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Dict(d) => format!(
                "{{{}}}",
                d.iter()
                    .map(|(k, v)| format!("{}: {}", python_string_repr(k), v.inline_repr()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Word(w) => format!("PdfKeyword(value={w})"),
            Self::Inline(..) => self.pdf(),
        }
    }
}
fn python_string_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = quote.to_string();
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c.is_control() || matches!(c, '\u{a0}' | '\u{ad}')) && (c as u32) <= 255 => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            _ => out.push(c),
        }
    }
    out.push(quote);
    out
}
pub(crate) fn python_bytes_repr(bytes: &[u8]) -> String {
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut out = format!("b{}", quote as char);
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            b if b == quote => {
                out.push('\\');
                out.push(b as char);
            }
            32..=126 => out.push(b as char),
            _ => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out.push(quote as char);
    out
}
pub fn decode_name(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '#' && i + 2 < chars.len() {
            if let (Some(a), Some(b)) = (chars[i + 1].to_digit(16), chars[i + 2].to_digit(16)) {
                out.push(((a * 16 + b) as u8) as char);
                i += 3;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1
    }
    out
}

pub(crate) fn canonical_name(s: &str) -> String {
    let mut current = s.to_owned();
    loop {
        let decoded = decode_name(&current);
        if decoded == current {
            return current;
        }
        current = decoded;
    }
}

#[derive(Debug)]
pub struct Operation {
    pub op: String,
    pub args: Vec<Token>,
}
pub struct Lexer<'a> {
    b: &'a [u8],
    p: usize,
    wrap_literal_octal: bool,
    pending: std::collections::VecDeque<Token>,
}

fn latin1_digit(byte: u8) -> bool {
    byte.is_ascii_digit() || matches!(byte, 0xb2 | 0xb3 | 0xb9)
}

fn latin1_string(bytes: &[u8]) -> String {
    bytes.iter().map(|&byte| char::from(byte)).collect()
}

// The reference recovers a whole keyword only if every fragment is a content
// keyword or numeric prefix. Partial recovery must leave unknown words intact.
fn composite_keyword(raw: &[u8]) -> Result<Option<std::collections::VecDeque<Token>>> {
    const KEYWORDS: &[&str] = &[
        "q", "Q", "cm", "w", "J", "j", "M", "d", "ri", "i", "gs", "m", "l", "c", "v", "y", "h",
        "re", "S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n", "W", "W*", "sh", "BI", "ID",
        "EI", "Do", "MP", "DP", "BMC", "BDC", "EMC", "BX", "EX", "BT", "ET", "Tc", "Tw", "Tz",
        "TL", "Tf", "Tr", "Ts", "Td", "TD", "Tm", "T*", "Tj", "TJ", "'", "\"", "CS", "cs", "SC",
        "SCN", "sc", "scn", "G", "g", "RG", "rg", "K", "k", "d0", "d1", "true", "false", "null",
    ];
    if !raw.iter().copied().any(latin1_digit)
        || KEYWORDS.iter().any(|keyword| raw == keyword.as_bytes())
    {
        return Ok(None);
    }
    let mut tokens = std::collections::VecDeque::new();
    let mut rest = raw;
    while !rest.is_empty() {
        let signed = usize::from(matches!(rest[0], b'+' | b'-'));
        let mut end = signed;
        while rest.get(end).is_some_and(|byte| latin1_digit(*byte)) {
            end += 1;
        }
        if rest.get(end) == Some(&b'.') {
            end += 1;
            while rest.get(end).is_some_and(|byte| latin1_digit(*byte)) {
                end += 1;
            }
        }
        if end > signed {
            let number = &rest[..end];
            if number != b"." && number != b"+." && number != b"-." {
                let text = latin1_string(number);
                if number.iter().any(|byte| !byte.is_ascii()) {
                    let message = if number.contains(&b'.') {
                        format!("could not convert string to float: '{text}'")
                    } else {
                        format!("invalid literal for int() with base 10: '{text}'")
                    };
                    return Err(Error::new("ValueError", message));
                }
                tokens.push_back(if number.contains(&b'.') {
                    Token::Real(text.parse().ok().ok_or_else(|| {
                        Error::new(
                            "ValueError",
                            format!("could not convert string to float: '{text}'"),
                        )
                    })?)
                } else {
                    text.parse::<i64>()
                        .map(Token::Int)
                        .unwrap_or_else(|_| Token::Real(text.parse().unwrap_or(f64::INFINITY)))
                });
                rest = &rest[end..];
                continue;
            }
            rest = &rest[1..];
            continue;
        }
        // A bare dot (and the dot in a sign-dot prefix) is discarded one byte
        // at a time by Python's _parse_numeric_prefix before keyword matching.
        if rest[0] == b'.' {
            rest = &rest[1..];
            continue;
        }
        let Some(keyword) = KEYWORDS
            .iter()
            .filter(|key| rest.starts_with(key.as_bytes()))
            .max_by_key(|key| key.len())
        else {
            return Ok(None);
        };
        tokens.push_back(Token::Word((*keyword).into()));
        rest = &rest[keyword.len()..];
    }
    Ok((tokens.len() > 1).then_some(tokens))
}
fn ws(b: u8) -> bool {
    b" \t\r\n\x0c\0".contains(&b)
}
fn delim(b: u8) -> bool {
    ws(b) || b"()<>[]{}/%".contains(&b)
}
impl<'a> Lexer<'a> {
    // Type3 bounds follow MuPDF's SVG interpreter, whose literal bytes wrap.
    // Page content follows the Python ContentStreamTokenizer's strict append.
    pub fn for_type3(b: &'a [u8]) -> Self {
        Self {
            wrap_literal_octal: true,
            ..Self::new(b)
        }
    }
    pub fn new(b: &'a [u8]) -> Self {
        Self {
            b,
            p: 0,
            wrap_literal_octal: false,
            pending: Default::default(),
        }
    }
    fn skip(&mut self) {
        loop {
            while self.p < self.b.len() && ws(self.b[self.p]) {
                self.p += 1
            }
            if self.p < self.b.len() && self.b[self.p] == b'%' {
                while self.p < self.b.len() && !b"\r\n".contains(&self.b[self.p]) {
                    self.p += 1
                }
            } else {
                break;
            }
        }
    }
    fn token(&mut self, depth: usize) -> Result<Option<Token>> {
        if depth > 128 {
            return Err(Error::new(
                "TokenizerError",
                "PDF token nesting exceeded max depth 128",
            ));
        }
        if let Some(token) = self.pending.pop_front() {
            return Ok(Some(token));
        }
        // The reference consumes a decimal prefix, including a sign or dot
        // without digits (which it discards), before looking for a keyword.
        loop {
            self.skip();
            let start = self.p;
            let mut end = start;
            if self.b.get(end).is_some_and(|b| b"+-".contains(b)) {
                end += 1;
            }
            while self.b.get(end).is_some_and(u8::is_ascii_digit) {
                end += 1;
            }
            if self.b.get(end) == Some(&b'.') {
                end += 1;
                while self.b.get(end).is_some_and(u8::is_ascii_digit) {
                    end += 1;
                }
            }
            if end == start {
                break;
            }
            self.p = end;
            let text = std::str::from_utf8(&self.b[start..end]).unwrap();
            if matches!(text, "+" | "-" | "." | "+." | "-.") {
                continue;
            }
            return Ok(Some(if !text.contains('.') {
                text.parse::<i64>()
                    .map(Token::Int)
                    .unwrap_or_else(|_| Token::Real(text.parse().unwrap()))
            } else {
                Token::Real(text.parse().unwrap())
            }));
        }
        self.skip();
        if self.p == self.b.len() {
            return Ok(None);
        }
        let c = self.b[self.p];
        self.p += 1;
        Ok(Some(match c {
            b']' | b'>' | b'}' if !self.wrap_literal_octal => {
                return Err(Error::new(
                    "TokenizerError",
                    format!("Unexpected delimiter b'{}'.", c as char),
                ));
            }
            b'/' => {
                let start = self.p;
                while self.p < self.b.len() && !delim(self.b[self.p]) {
                    self.p += 1
                }
                Token::Name(self.b[start..self.p].iter().map(|&x| x as char).collect())
            }
            b'(' => {
                let mut v = Vec::new();
                let mut nesting = 1;
                while self.p < self.b.len() {
                    let x = self.b[self.p];
                    self.p += 1;
                    match x {
                        b'(' => {
                            nesting += 1;
                            v.push(x)
                        }
                        b')' => {
                            nesting -= 1;
                            if nesting == 0 {
                                break;
                            }
                            v.push(x)
                        }
                        b'\\' => {
                            if self.p == self.b.len() {
                                break;
                            }
                            let x = self.b[self.p];
                            self.p += 1;
                            match x {
                                b'n' => v.push(10),
                                b'r' => v.push(13),
                                b't' => v.push(9),
                                b'b' => v.push(8),
                                b'f' => v.push(12),
                                b'\r' => {
                                    if self.b.get(self.p) == Some(&b'\n') {
                                        self.p += 1
                                    }
                                }
                                b'\n' => {}
                                b'0'..=b'7' => {
                                    let mut n = (x - b'0') as u16;
                                    for _ in 0..2 {
                                        if let Some(&x @ b'0'..=b'7') = self.b.get(self.p) {
                                            n = n * 8 + (x - b'0') as u16;
                                            self.p += 1
                                        } else {
                                            break;
                                        }
                                    }
                                    if n > 255 && !self.wrap_literal_octal {
                                        return Err(Error::new(
                                            "ValueError",
                                            "byte must be in range(0, 256)",
                                        ));
                                    }
                                    v.push(n as u8)
                                }
                                _ => v.push(x),
                            }
                        }
                        b'\r' => v.push(x),
                        _ => v.push(x),
                    }
                }
                if nesting != 0 {
                    return Err(Error::new("TokenizerError", "Unterminated literal string."));
                }
                Token::Str(v, false)
            }
            b'[' => {
                let mut v = Vec::new();
                loop {
                    self.skip();
                    if self.b.get(self.p) == Some(&b']') {
                        self.p += 1;
                        break;
                    }
                    if let Some(t) = self.token(depth + 1)? {
                        v.push(t)
                    } else {
                        return Err(Error::new("TokenizerError", "Unterminated array"));
                    }
                }
                Token::Array(v)
            }
            b'<' => {
                if self.b.get(self.p) == Some(&b'<') {
                    self.p += 1;
                    let mut v = Vec::new();
                    loop {
                        self.skip();
                        if self.b.get(self.p..self.p + 2) == Some(b">>") {
                            self.p += 2;
                            break;
                        }
                        let k = self.token(depth + 1)?.ok_or_else(|| {
                            Error::new("TokenizerError", "Unterminated dictionary")
                        })?;
                        let k = k.name()?.to_owned();
                        let x = self.token(depth + 1)?.ok_or_else(|| {
                            Error::new("TokenizerError", "Missing dictionary value")
                        })?;
                        v.push((k, x))
                    }
                    Token::Dict(v)
                } else {
                    let mut hex = Vec::new();
                    while self.p < self.b.len() && self.b[self.p] != b'>' {
                        let x = self.b[self.p];
                        self.p += 1;
                        if !ws(x) {
                            hex.push(x)
                        }
                    }
                    if self.p < self.b.len() {
                        self.p += 1
                    }
                    if hex.len() % 2 != 0 {
                        hex.push(b'0')
                    }
                    let mut out = Vec::new();
                    for h in hex.chunks(2) {
                        let s = std::str::from_utf8(h).map_err(|_| {
                            Error::new("TokenizerError", "Invalid hexadecimal string")
                        })?;
                        out.push(u8::from_str_radix(s, 16).map_err(|_| {
                            Error::new("TokenizerError", "Invalid hexadecimal string")
                        })?)
                    }
                    Token::Str(out, true)
                }
            }
            _ => {
                let start = self.p - 1;
                while self.p < self.b.len() && !delim(self.b[self.p]) {
                    self.p += 1
                }
                let s = String::from_utf8_lossy(&self.b[start..self.p]).to_string();
                if let Ok(n) = s.parse::<i64>() {
                    Token::Int(n)
                } else if let Ok(n) = s.parse::<f64>() {
                    Token::Real(n)
                } else {
                    if let Some(mut tokens) = composite_keyword(&self.b[start..self.p])? {
                        let first = tokens.pop_front().unwrap();
                        self.pending = tokens;
                        first
                    } else {
                        Token::Word(s)
                    }
                }
            }
        }))
    }
    fn inline_delimited(&self, pos: usize, keyword: &[u8]) -> bool {
        self.b.get(pos..pos + keyword.len()) == Some(keyword)
            && (pos == 0 || ws(self.b[pos - 1]))
            && self.b.get(pos + keyword.len()).is_none_or(|b| ws(*b))
    }
    fn inline_tail(&self, mut pos: usize) -> bool {
        loop {
            while self.b.get(pos).is_some_and(|b| ws(*b)) {
                pos += 1;
            }
            if self.b.get(pos) == Some(&b'%') {
                while self.b.get(pos).is_some_and(|b| !b"\r\n".contains(b)) {
                    pos += 1;
                }
            } else {
                break;
            }
        }
        if pos == self.b.len() || self.b.get(pos) == Some(&b'/') {
            return true;
        }
        let start = pos;
        while self.b.get(pos).is_some_and(|b| !delim(*b)) {
            pos += 1;
        }
        let token = &self.b[start..pos];
        [
            b"Q".as_slice(),
            b"q",
            b"cm",
            b"Do",
            b"BT",
            b"ET",
            b"W",
            b"W*",
            b"S",
            b"s",
            b"f",
            b"F",
            b"f*",
            b"B",
            b"B*",
            b"b",
            b"b*",
            b"n",
            b"BI",
            b"ID",
            b"EMC",
            b"BDC",
            b"Tj",
            b"TJ",
            b"Td",
            b"TD",
            b"Tm",
            b"Tf",
            b"TL",
            b"Tc",
            b"Tw",
            b"Tz",
            b"Ts",
            b"Tr",
            b"gs",
        ]
        .contains(&token)
    }
    pub fn operation(&mut self) -> Result<Option<Operation>> {
        let mut args = Vec::new();
        while let Some(t) = self.token(0)? {
            if let Token::Word(op) = t {
                if op == "BI" {
                    let mut dict: Vec<(std::rc::Rc<str>, Token)> = Vec::new();
                    let mut slots = std::collections::HashMap::<std::rc::Rc<str>, usize>::new();
                    loop {
                        let k = self.token(0)?.ok_or_else(|| {
                            Error::new("TokenizerError", "Unterminated inline image")
                        })?;
                        if matches!(&k,Token::Word(x) if x=="ID") {
                            break;
                        }
                        let key = match k {
                            Token::Name(k) | Token::Word(k) => k,
                            _ => {
                                return Err(Error::new(
                                    "TokenizerError",
                                    "Inline image dictionary key missing",
                                ));
                            }
                        };
                        let val = self.token(0)?.ok_or_else(|| {
                            Error::new("TokenizerError", "Inline image missing value")
                        })?;
                        if let Some(&slot) = slots.get(key.as_str()) {
                            dict[slot].1 = val;
                        } else {
                            let key: std::rc::Rc<str> = key.into();
                            slots.insert(key.clone(), dict.len());
                            dict.push((key, val));
                        }
                    }
                    drop(slots);
                    let dict: Vec<(String, Token)> = dict.into_iter()
                        .map(|(key, value)| (key.to_string(), value)).collect();
                    // The Python tokenizer consumes exactly one byte after ID.
                    if self.b.get(self.p).is_some_and(|b| ws(*b)) {
                        self.p += 1;
                    }
                    let mut start = self.p;
                    let length = ["L", "Length"].iter().find_map(|key| {
                        dict.iter()
                            .find(|(k, _)| k == key)
                            .and_then(|(_, v)| match v {
                                Token::Int(n) if *n >= 0 => usize::try_from(*n).ok(),
                                _ => None,
                            })
                    });
                    let mut exact_end = None;
                    if let Some(length) = length {
                        if let Some(end) = start.checked_add(length).filter(|e| *e <= self.b.len())
                        {
                            self.p = end;
                            let mut probe = end;
                            while self.b.get(probe).is_some_and(|b| ws(*b)) {
                                probe += 1;
                            }
                            if probe == self.b.len() {
                                self.p = probe;
                                exact_end = Some(end);
                            } else if self.inline_delimited(probe, b"EI") {
                                self.p = probe + 2;
                                exact_end = Some(end);
                            } else {
                                // Reference keeps its advanced position on a failed
                                // declared-length probe before trying the EI heuristic.
                                start = end;
                            }
                        }
                    }
                    let end = if let Some(end) = exact_end {
                        end
                    } else {
                        let mut found = None;
                        while self.p + 1 < self.b.len() {
                            if self.inline_delimited(self.p, b"EI") && self.inline_tail(self.p + 2)
                            {
                                found = Some(self.p);
                                self.p += 2;
                                break;
                            }
                            self.p += 1;
                        }
                        found.ok_or_else(|| {
                            Error::new("TokenizerError", "Unterminated inline image data")
                        })?
                    };
                    return Ok(Some(Operation {
                        op: "INLINE_IMAGE".into(),
                        args: vec![Token::Inline(dict, self.b[start..end].to_vec())],
                    }));
                }
                return Ok(Some(Operation { op, args }));
            } else {
                args.push(t)
            }
        }
        Ok(None)
    }
    pub fn operations(&mut self) -> Result<Vec<Operation>> {
        let mut out = Vec::new();
        while let Some(operation) = self.operation()? {
            out.push(operation);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tokens_tests;
