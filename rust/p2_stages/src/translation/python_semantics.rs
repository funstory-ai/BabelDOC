use std::collections::HashMap;

use crate::shared::layout_helper::unicode::is_python_strip_whitespace;

#[derive(Clone, Debug, PartialEq)]
pub enum PythonJsonValue {
    Null,
    Bool(bool),
    Integer(String),
    Float(f64),
    String(String),
    Array(Vec<PythonJsonValue>),
    Object(Vec<(String, PythonJsonValue)>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JsonCompatError(pub String);

impl std::fmt::Display for JsonCompatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for JsonCompatError {}

pub fn clean_json_output(input: &str) -> String {
    let mut output = python_strip(input).to_owned();
    if output.starts_with("<json>") {
        output = output[6..].to_owned();
    }
    if output.ends_with("</json>") {
        output.truncate(output.len() - 7);
    }
    if output.starts_with("```json") {
        output = output[7..].to_owned();
    }
    if output.starts_with("```") {
        output = output[3..].to_owned();
    }
    if output.ends_with("```") {
        output.truncate(output.len() - 3);
    }
    python_strip(&output).to_owned()
}

pub fn python_strip(text: &str) -> &str {
    let start = text
        .char_indices()
        .find(|(_, ch)| !is_python_strip_whitespace(*ch))
        .map_or(text.len(), |(index, _)| index);
    let end = text
        .char_indices()
        .rev()
        .find(|(_, ch)| !is_python_strip_whitespace(*ch))
        .map_or(start, |(index, ch)| index + ch.len_utf8());
    &text[start..end]
}

pub fn python_isspace(text: &str) -> bool {
    !text.is_empty() && text.chars().all(is_python_strip_whitespace)
}

pub fn template_substitute(
    template: &str,
    values: &HashMap<&str, &str>,
) -> Result<String, JsonCompatError> {
    let chars: Vec<char> = template.chars().collect();
    let mut output = String::with_capacity(template.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '$' {
            output.push(chars[i]);
            i += 1;
            continue;
        }
        if chars.get(i + 1) == Some(&'$') {
            output.push('$');
            i += 2;
            continue;
        }
        let (start, end) = if chars.get(i + 1) == Some(&'{') {
            let start = i + 2;
            let Some(relative_end) = chars[start..].iter().position(|ch| *ch == '}') else {
                return Err(JsonCompatError("invalid Template placeholder".into()));
            };
            (start, start + relative_end)
        } else {
            let start = i + 1;
            let Some(first) = chars.get(start) else {
                return Err(JsonCompatError("invalid Template placeholder".into()));
            };
            if !first.is_ascii_alphabetic() && *first != '_' {
                return Err(JsonCompatError("invalid Template placeholder".into()));
            }
            let mut end = start + 1;
            while chars
                .get(end)
                .is_some_and(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            {
                end += 1;
            }
            (start, end)
        };
        let key: String = chars[start..end].iter().collect();
        let value = values
            .get(key.as_str())
            .ok_or_else(|| JsonCompatError(format!("missing Template key: {key}")))?;
        output.push_str(value);
        i = if chars.get(i + 1) == Some(&'{') {
            end + 1
        } else {
            end
        };
    }
    Ok(output)
}

pub fn format_named_template(
    template: &str,
    values: &HashMap<&str, &str>,
) -> Result<String, JsonCompatError> {
    let chars: Vec<char> = template.chars().collect();
    let mut output = String::with_capacity(template.len());
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '{' if chars.get(i + 1) == Some(&'{') => {
                output.push('{');
                i += 2;
            }
            '}' if chars.get(i + 1) == Some(&'}') => {
                output.push('}');
                i += 2;
            }
            '{' => {
                let start = i + 1;
                let Some(relative_end) = chars[start..].iter().position(|ch| *ch == '}') else {
                    return Err(JsonCompatError("unmatched '{' in format string".into()));
                };
                let end = start + relative_end;
                let key: String = chars[start..end].iter().collect();
                let value = values
                    .get(key.as_str())
                    .ok_or_else(|| JsonCompatError(format!("missing format key: {key}")))?;
                output.push_str(value);
                i = end + 1;
            }
            '}' => return Err(JsonCompatError("unmatched '}' in format string".into())),
            ch => {
                output.push(ch);
                i += 1;
            }
        }
    }
    Ok(output)
}

pub fn levenshtein_distance(left: &str, right: &str) -> usize {
    if left == right {
        return 0;
    }
    let (left, right) = trim_common_edges(left, right);
    let left_len = left.chars().count();
    let right_len = right.chars().count();
    if left_len == 0 {
        return right_len;
    }
    if right_len == 0 {
        return left_len;
    }

    let (columns, rows, column_len) = if left_len <= right_len {
        (left, right, left_len)
    } else {
        (right, left, right_len)
    };
    let columns: Vec<char> = columns.chars().collect();
    let mut row: Vec<usize> = (0..=column_len).collect();
    for a in rows.chars() {
        let mut diagonal = row[0];
        row[0] += 1;
        for (j, b) in columns.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (row[j] + 1)
                .min(above + 1)
                .min(diagonal + usize::from(a != *b));
            diagonal = above;
        }
    }
    row[column_len]
}

pub(crate) fn levenshtein_distance_at_most(
    left: &str,
    right: &str,
    max_distance: usize,
) -> Option<usize> {
    if left == right {
        return Some(0);
    }
    let (left, right) = trim_common_edges(left, right);
    let left_len = left.chars().count();
    let right_len = right.chars().count();
    if left_len.abs_diff(right_len) > max_distance {
        return None;
    }

    let (columns, rows, column_len) = if left_len <= right_len {
        (left, right, left_len)
    } else {
        (right, left, right_len)
    };
    if column_len == 0 {
        return Some(rows.chars().count());
    }
    let columns: Vec<char> = columns.chars().collect();

    let too_far = max_distance.saturating_add(1);
    let mut previous = vec![too_far; column_len + 1];
    for (distance, cell) in previous
        .iter_mut()
        .take(column_len.min(max_distance) + 1)
        .enumerate()
    {
        *cell = distance;
    }
    let mut current = vec![too_far; column_len + 1];

    for (row_index, a) in rows.chars().enumerate() {
        let i = row_index + 1;
        let start = i.saturating_sub(max_distance);
        let end = column_len.min(i.saturating_add(max_distance));
        let mut row_min = too_far;
        if start == 0 {
            current[0] = i;
            row_min = i;
        } else {
            current[start - 1] = too_far;
        }

        for j in start.max(1)..=end {
            let distance = previous[j]
                .saturating_add(1)
                .min(current[j - 1].saturating_add(1))
                .min(previous[j - 1].saturating_add(usize::from(a != columns[j - 1])))
                .min(too_far);
            current[j] = distance;
            row_min = row_min.min(distance);
        }
        if end < column_len {
            current[end + 1] = too_far;
        }
        if row_min > max_distance {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }

    let distance = previous[column_len];
    (distance <= max_distance).then_some(distance)
}

fn trim_common_edges<'a>(left: &'a str, right: &'a str) -> (&'a str, &'a str) {
    let prefix_len = left
        .char_indices()
        .zip(right.char_indices())
        .take_while(|((_, left_char), (_, right_char))| left_char == right_char)
        .last()
        .map_or(0, |((index, ch), _)| index + ch.len_utf8());
    let left = &left[prefix_len..];
    let right = &right[prefix_len..];
    let suffix_len = left
        .char_indices()
        .rev()
        .zip(right.char_indices().rev())
        .take_while(|((_, left_char), (_, right_char))| left_char == right_char)
        .map(|((_, ch), _)| ch.len_utf8())
        .sum::<usize>();
    (
        &left[..left.len() - suffix_len],
        &right[..right.len() - suffix_len],
    )
}

pub fn python_float_str(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value == f64::INFINITY {
        return "inf".into();
    }
    if value == f64::NEG_INFINITY {
        return "-inf".into();
    }
    if value == 0.0 && value.is_sign_negative() {
        return "-0.0".into();
    }
    let rendered = value.to_string();
    if (value.abs() >= 1e16 || (value != 0.0 && value.abs() < 1e-4)) && !rendered.contains('e') {
        return decimal_to_exponent(&rendered);
    }
    let mut rendered = rendered;
    if rendered.contains('e') {
        rendered = normalize_exponent(rendered);
    } else if !rendered.contains('.') {
        rendered.push_str(".0");
    }
    rendered
}

pub fn python_number_str(value: &PythonJsonValue) -> Option<String> {
    match value {
        PythonJsonValue::Integer(value) => Some(value.clone()),
        PythonJsonValue::Float(value) => Some(python_float_str(*value)),
        _ => None,
    }
}

fn normalize_exponent(value: String) -> String {
    let (mantissa, exponent) = value.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let abs_exponent = exponent.unsigned_abs();
    let exponent_text = if abs_exponent < 10 {
        format!("0{abs_exponent}")
    } else {
        abs_exponent.to_string()
    };
    format!(
        "{mantissa}e{}{exponent_text}",
        if exponent >= 0 { "+" } else { "-" }
    )
}

fn decimal_to_exponent(value: &str) -> String {
    let negative = value.starts_with('-');
    let unsigned = value.strip_prefix('-').unwrap_or(value);
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let digits = format!("{whole}{fraction}");
    let first_nonzero = digits.find(|c| c != '0').unwrap_or(digits.len());
    if first_nonzero == digits.len() {
        return if negative { "-0.0" } else { "0.0" }.into();
    }
    let exponent = whole.len() as i32 - first_nonzero as i32 - 1;
    let significant = digits[first_nonzero..].trim_end_matches('0').to_owned();
    if significant.len() == 1 {
        let abs_exponent = exponent.unsigned_abs();
        let exponent_text = if abs_exponent < 10 {
            format!("0{abs_exponent}")
        } else {
            abs_exponent.to_string()
        };
        return format!(
            "{}{significant}e{}{exponent_text}",
            if negative { "-" } else { "" },
            if exponent >= 0 { "+" } else { "-" }
        );
    }
    let mantissa = format!("{}.{}", &significant[..1], &significant[1..]);
    format!(
        "{}{mantissa}e{}{exponent}",
        if negative { "-" } else { "" },
        if exponent >= 0 { "+" } else { "" }
    )
}

pub fn json_loads(input: &str) -> Result<PythonJsonValue, JsonCompatError> {
    let mut parser = JsonParser {
        input,
        offset: 0,
        depth: 0,
    };
    let value = parser.value()?;
    parser.whitespace();
    if parser.offset != input.len() {
        return Err(parser.error("extra data"));
    }
    Ok(value)
}

struct JsonParser<'a> {
    input: &'a str,
    offset: usize,
    depth: usize,
}

impl JsonParser<'_> {
    fn error(&self, message: &str) -> JsonCompatError {
        JsonCompatError(format!("{message} at byte {}", self.offset))
    }
    fn whitespace(&mut self) {
        while self.input[self.offset..]
            .chars()
            .next()
            .is_some_and(|c| matches!(c, ' ' | '\t' | '\n' | '\r'))
        {
            self.offset += self.input[self.offset..].chars().next().unwrap().len_utf8();
        }
    }
    fn value(&mut self) -> Result<PythonJsonValue, JsonCompatError> {
        self.whitespace();
        if self.depth > 1000 {
            return Err(self.error("maximum recursion depth exceeded"));
        }
        match self.input.as_bytes().get(self.offset).copied() {
            Some(b'n') => {
                self.literal("null")?;
                Ok(PythonJsonValue::Null)
            }
            Some(b't') => {
                self.literal("true")?;
                Ok(PythonJsonValue::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(PythonJsonValue::Bool(false))
            }
            Some(b'"') => self.string().map(PythonJsonValue::String),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
            Some(b'-' | b'0'..=b'9' | b'N' | b'I') => self.number(),
            _ => Err(self.error("expecting value")),
        }
    }
    fn literal(&mut self, literal: &str) -> Result<(), JsonCompatError> {
        if self.input[self.offset..].starts_with(literal) {
            self.offset += literal.len();
            Ok(())
        } else {
            Err(self.error("invalid literal"))
        }
    }
    fn string(&mut self) -> Result<String, JsonCompatError> {
        let start = self.offset;
        self.offset += 1;
        let bytes = self.input.as_bytes();
        let mut escaped = false;
        while self.offset < bytes.len() {
            let byte = bytes[self.offset];
            if escaped {
                escaped = false;
                self.offset += 1;
                continue;
            }
            match byte {
                b'\\' => {
                    escaped = true;
                    self.offset += 1;
                }
                b'"' => {
                    self.offset += 1;
                    return serde_json::from_str(&self.input[start..self.offset])
                        .map_err(|error| JsonCompatError(error.to_string()));
                }
                0..=0x1f => return Err(self.error("control character in string")),
                _ => {
                    let ch = self.input[self.offset..]
                        .chars()
                        .next()
                        .ok_or_else(|| self.error("unterminated string"))?;
                    self.offset += ch.len_utf8();
                }
            }
        }
        Err(self.error("unterminated string"))
    }
    fn array(&mut self) -> Result<PythonJsonValue, JsonCompatError> {
        self.offset += 1;
        self.depth += 1;
        self.whitespace();
        let mut values = Vec::new();
        if self.input[self.offset..].starts_with(']') {
            self.offset += 1;
            self.depth -= 1;
            return Ok(PythonJsonValue::Array(values));
        }
        loop {
            values.push(self.value()?);
            self.whitespace();
            if self.input[self.offset..].starts_with(']') {
                self.offset += 1;
                self.depth -= 1;
                break;
            }
            if !self.input[self.offset..].starts_with(',') {
                return Err(self.error("expecting ',' delimiter"));
            }
            self.offset += 1;
        }
        Ok(PythonJsonValue::Array(values))
    }
    fn object(&mut self) -> Result<PythonJsonValue, JsonCompatError> {
        self.offset += 1;
        self.depth += 1;
        self.whitespace();
        let mut entries: Vec<(String, PythonJsonValue)> = Vec::new();
        if self.input[self.offset..].starts_with('}') {
            self.offset += 1;
            self.depth -= 1;
            return Ok(PythonJsonValue::Object(entries));
        }
        loop {
            self.whitespace();
            if !self.input[self.offset..].starts_with('"') {
                return Err(self.error("object keys must be strings"));
            }
            let key = self.string()?;
            self.whitespace();
            if !self.input[self.offset..].starts_with(':') {
                return Err(self.error("expecting ':' delimiter"));
            }
            self.offset += 1;
            let value = self.value()?;
            if let Some((_, old)) = entries.iter_mut().find(|(existing, _)| existing == &key) {
                *old = value;
            } else {
                entries.push((key, value));
            }
            self.whitespace();
            if self.input[self.offset..].starts_with('}') {
                self.offset += 1;
                self.depth -= 1;
                break;
            }
            if !self.input[self.offset..].starts_with(',') {
                return Err(self.error("expecting ',' delimiter"));
            }
            self.offset += 1;
        }
        Ok(PythonJsonValue::Object(entries))
    }
    fn number(&mut self) -> Result<PythonJsonValue, JsonCompatError> {
        for (literal, value) in [
            ("-Infinity", f64::NEG_INFINITY),
            ("Infinity", f64::INFINITY),
            ("NaN", f64::NAN),
        ] {
            if self.input[self.offset..].starts_with(literal) {
                self.offset += literal.len();
                return Ok(PythonJsonValue::Float(value));
            }
        }
        let start = self.offset;
        let bytes = self.input.as_bytes();
        if bytes.get(self.offset) == Some(&b'-') {
            self.offset += 1;
        }
        if bytes.get(self.offset) == Some(&b'0') {
            self.offset += 1;
        } else {
            let digit_start = self.offset;
            while bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
                self.offset += 1;
            }
            if digit_start == self.offset {
                return Err(self.error("invalid number"));
            }
        }
        if bytes.get(self.offset) == Some(&b'.') {
            self.offset += 1;
            let digit_start = self.offset;
            while bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
                self.offset += 1;
            }
            if digit_start == self.offset {
                return Err(self.error("invalid number"));
            }
        }
        if bytes
            .get(self.offset)
            .is_some_and(|b| matches!(b, b'e' | b'E'))
        {
            self.offset += 1;
            if bytes
                .get(self.offset)
                .is_some_and(|b| matches!(b, b'+' | b'-'))
            {
                self.offset += 1;
            }
            let digit_start = self.offset;
            while bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
                self.offset += 1;
            }
            if digit_start == self.offset {
                return Err(self.error("invalid number"));
            }
        }
        let token = &self.input[start..self.offset];
        if token.contains(['.', 'e', 'E']) {
            let value = token.parse().map_err(|_| self.error("invalid number"))?;
            Ok(PythonJsonValue::Float(value))
        } else {
            let digits = token.trim_start_matches('-');
            if digits.len() > 4300 {
                return Err(
                    self.error("Exceeds the limit (4300 digits) for integer string conversion")
                );
            }
            Ok(PythonJsonValue::Integer(token.to_owned()))
        }
    }
}

pub fn json_dumps(value: &PythonJsonValue) -> String {
    let mut out = String::new();
    dump_value(value, 0, &mut out);
    out
}

fn dump_value(value: &PythonJsonValue, depth: usize, out: &mut String) {
    match value {
        PythonJsonValue::Null => out.push_str("null"),
        PythonJsonValue::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        PythonJsonValue::Integer(value) => out.push_str(value),
        PythonJsonValue::Float(value) if value.is_nan() => out.push_str("NaN"),
        PythonJsonValue::Float(value) if *value == f64::INFINITY => out.push_str("Infinity"),
        PythonJsonValue::Float(value) if *value == f64::NEG_INFINITY => out.push_str("-Infinity"),
        PythonJsonValue::Float(value) => out.push_str(&python_float_str(*value)),
        PythonJsonValue::String(value) => dump_string(value, out),
        PythonJsonValue::Array(values) => {
            if values.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (index, value) in values.iter().enumerate() {
                indent(depth + 1, out);
                dump_value(value, depth + 1, out);
                out.push_str(if index + 1 == values.len() {
                    "\n"
                } else {
                    ",\n"
                });
            }
            indent(depth, out);
            out.push(']');
        }
        PythonJsonValue::Object(values) => {
            if values.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (index, (key, value)) in values.iter().enumerate() {
                indent(depth + 1, out);
                dump_string(key, out);
                out.push_str(": ");
                dump_value(value, depth + 1, out);
                out.push_str(if index + 1 == values.len() {
                    "\n"
                } else {
                    ",\n"
                });
            }
            indent(depth, out);
            out.push('}');
        }
    }
}
fn indent(depth: usize, out: &mut String) {
    for _ in 0..depth * 2 {
        out.push(' ');
    }
}
fn dump_string(value: &str, out: &mut String) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch <= '\u{1f}' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod python_semantics_tests;
