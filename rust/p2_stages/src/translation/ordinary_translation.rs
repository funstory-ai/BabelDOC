//! Pure ordinary-translation planning and tracking values.
//!
//! Translator calls and translator-supplied regular expressions remain in the
//! bridge's I/O-bound adapter; the data shape and deterministic request policy
//! live here so the LLM-only fallback can reuse them.

use super::python_semantics::{PythonJsonValue, json_dumps};
use super::token_count::count_tokens;

/// Narrow surface for translator-owned Python regular expressions. The caller
/// invokes this only from a translation job after it has entered the I/O pool.
pub trait PlaceholderRegex {
    type Error;
    fn compile(&mut self, pattern: &str, ignore_case: bool) -> Result<u64, Self::Error>;
    fn matches(&mut self, pattern: u64, text: &str) -> Result<bool, Self::Error>;
    fn find_all(&mut self, pattern: u64, text: &str) -> Result<Vec<RegexMatch>, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegexMatch {
    pub full: String,
    pub start: usize,
    pub end: usize,
    pub groups: Vec<Option<(usize, usize)>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaceholderPatterns {
    pub formula: String,
    pub rich_left: String,
    pub rich_right: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaceholderReturn {
    Text(String),
    Tuple { token: String, pattern: String },
}

/// `ILTranslator.__init__` indexes the return value with `[1]` without first
/// checking whether it is a tuple. Preserve the BaseTranslator string shape.
pub fn legacy_hallucination_pattern(value: &PlaceholderReturn) -> Result<String, &'static str> {
    match value {
        PlaceholderReturn::Tuple { pattern, .. } => Ok(pattern.clone()),
        PlaceholderReturn::Text(text) => text
            .chars()
            .nth(1)
            .map(|ch| ch.to_string())
            .ok_or("string index out of range"),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormulaInput {
    pub id: i64,
    pub composition_index: usize,
    pub token: String,
    pub pattern: String,
    pub source_chars: String,
    pub hint_source: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RichInput {
    pub id: i64,
    pub composition_index: usize,
    pub left: String,
    pub right: String,
    pub left_pattern: String,
    pub right_pattern: String,
    pub source_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputPlaceholder {
    Formula(FormulaInput),
    Rich(RichInput),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslateInput {
    pub text: String,
    pub placeholders: Vec<InputPlaceholder>,
    pub original_tokens: Vec<(String, usize)>,
}

/// Implement Python's anchored, case-insensitive collision test for a proposed
/// token. IDs only advance when the translator's own regex recognizes a
/// collision in the original paragraph.
pub fn formula_placeholder<R: PlaceholderRegex>(
    regex: &mut R,
    paragraph: &str,
    mut id: i64,
    composition_index: usize,
    mut make: impl FnMut(i64) -> Result<(String, String), R::Error>,
    source_chars: String,
    hint_source: bool,
) -> Result<FormulaInput, R::Error> {
    loop {
        let (token, pattern) = make(id)?;
        let compiled = regex.compile(&pattern, true)?;
        if !regex.matches(compiled, paragraph)? {
            return Ok(FormulaInput {
                id,
                composition_index,
                token,
                pattern,
                source_chars,
                hint_source,
            });
        }
        id += 1;
    }
}

pub fn rich_placeholder<R: PlaceholderRegex>(
    regex: &mut R,
    paragraph: &str,
    mut id: i64,
    composition_index: usize,
    mut make: impl FnMut(i64) -> Result<(String, String, String, String), R::Error>,
    source_text: String,
) -> Result<RichInput, R::Error> {
    loop {
        let (left, right, left_pattern, right_pattern) = make(id)?;
        let combined = format!("{left_pattern}|{right_pattern}");
        let compiled = regex.compile(&combined, true)?;
        if !regex.matches(compiled, paragraph)? {
            return Ok(RichInput {
                id,
                composition_index,
                left,
                right,
                left_pattern,
                right_pattern,
                source_text,
            });
        }
        id += 1;
    }
}

pub fn original_placeholder_tokens<R: PlaceholderRegex>(
    regex: &mut R,
    patterns: &PlaceholderPatterns,
    text: &str,
) -> Result<Vec<(String, usize)>, R::Error> {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for pattern in [&patterns.formula, &patterns.rich_left, &patterns.rich_right] {
        let id = regex.compile(pattern, true)?;
        for found in regex.find_all(id, text)? {
            *counts.entry(found.full).or_default() += 1;
        }
    }
    Ok(counts.into_iter().collect())
}

pub fn prepare_translated_text(
    parts: &[String],
    placeholders: Vec<InputPlaceholder>,
    original_tokens: Vec<(String, usize)>,
) -> TranslateInput {
    TranslateInput {
        text: parts.concat(),
        placeholders,
        original_tokens,
    }
}

pub fn original_token_counts(input: &TranslateInput) -> std::collections::HashMap<String, usize> {
    input.original_tokens.iter().cloned().collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParsedPart {
    Text(String),
    Formula {
        composition_index: usize,
    },
    Rich {
        composition_index: usize,
        text: String,
        matches_source: bool,
    },
}

fn byte_offset(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map_or(text.len(), |(offset, _)| offset)
}

fn range_text(text: &str, range: (usize, usize)) -> &str {
    &text[byte_offset(text, range.0)..byte_offset(text, range.1)]
}

fn strip_hallucinated<R: PlaceholderRegex>(
    regex: &mut R,
    text: &str,
    own_patterns: &[String],
    hallucination_patterns: &[u64],
    allowed: &std::collections::HashSet<String>,
    removed: &mut Vec<(String, usize)>,
) -> Result<String, R::Error> {
    let mut output = text.to_owned();
    // Python 0.6.4 joins all translator-owned patterns into one alternation and
    // runs a single re.sub before checking the three hallucination patterns.
    if !own_patterns.is_empty() {
        let combined = regex.compile(&own_patterns.join("|"), true)?;
        output = remove_matches(regex, combined, &output, None, removed)?;
    }
    for pattern in hallucination_patterns {
        output = remove_matches(regex, *pattern, &output, Some(allowed), removed)?;
    }
    Ok(output)
}

fn remove_matches<R: PlaceholderRegex>(
    regex: &mut R,
    pattern: u64,
    text: &str,
    allowed: Option<&std::collections::HashSet<String>>,
    removed: &mut Vec<(String, usize)>,
) -> Result<String, R::Error> {
    let found = regex.find_all(pattern, text)?;
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for item in found {
        let start = byte_offset(text, item.start);
        let end = byte_offset(text, item.end);
        if start < cursor || end < start || end > text.len() {
            continue;
        }
        output.push_str(&text[cursor..start]);
        let token = item.full;
        let keep = allowed.is_some_and(|allowed| allowed.contains(&token));
        if keep {
            output.push_str(&token);
        } else if allowed.is_some() && !token.is_empty() {
            if let Some((_, count)) = removed.iter_mut().find(|(value, _)| value == &token) {
                *count += 1;
            } else {
                removed.push((token, 1));
            }
        }
        cursor = end;
    }
    output.push_str(&text[cursor..]);
    Ok(output)
}

fn placeholder_cleanup_data(
    input: &TranslateInput,
) -> (Vec<String>, std::collections::HashSet<String>) {
    let mut patterns = Vec::new();
    let mut allowed = input
        .original_tokens
        .iter()
        .map(|(token, _)| token.clone())
        .collect::<std::collections::HashSet<_>>();
    for placeholder in &input.placeholders {
        match placeholder {
            InputPlaceholder::Formula(formula) => {
                patterns.push(format!("({})", formula.pattern));
                allowed.insert(formula.token.clone());
            }
            InputPlaceholder::Rich(rich) => {
                patterns.extend([
                    format!("({})", rich.left_pattern),
                    format!("({})", rich.right_pattern),
                ]);
                allowed.extend([rich.left.clone(), rich.right.clone()]);
            }
        }
    }
    (patterns, allowed)
}

/// Apply 0.6.4's `remove_placeholder` body to a rich-text match after the
/// caller has decided to replace the source composition with translated text.
pub fn remove_placeholder_content<R: PlaceholderRegex>(
    regex: &mut R,
    input: &TranslateInput,
    text: &str,
    hallucination_patterns: &[u64],
    removed: &mut Vec<(String, usize)>,
) -> Result<String, R::Error> {
    let (patterns, allowed) = placeholder_cleanup_data(input);
    strip_hallucinated(
        regex,
        text,
        &patterns,
        hallucination_patterns,
        &allowed,
        removed,
    )
}

pub fn rich_text_matches_source(output: &str, source: &str) -> bool {
    output.replace(' ', "") == source.replace(' ', "")
}

/// Split the response using translator-owned patterns. The regex bridge reports
/// Python codepoint spans, which are converted to UTF-8 byte offsets here.
pub fn parse_output<R: PlaceholderRegex>(
    regex: &mut R,
    input: &TranslateInput,
    output: &str,
    hallucination_patterns: &[u64],
    removed: &mut Vec<(String, usize)>,
) -> Result<(Vec<ParsedPart>, bool), R::Error> {
    if input.placeholders.is_empty() {
        return Ok((vec![ParsedPart::Text(output.to_owned())], true));
    }
    let mut all_full = true;
    let mut patterns = Vec::new();
    let (token_patterns, allowed) = placeholder_cleanup_data(input);
    for item in &input.placeholders {
        match item {
            InputPlaceholder::Formula(formula) => {
                let pattern = regex.compile(&format!("({})", formula.pattern), true)?;
                if regex.find_all(pattern, output)?.is_empty() {
                    all_full = false;
                }
                patterns.push(format!("({})", formula.pattern));
            }
            InputPlaceholder::Rich(rich) => {
                let joined = format!("({}.*?{})", rich.left_pattern, rich.right_pattern);
                let pattern = regex.compile(&joined, true)?;
                if regex.find_all(pattern, output)?.is_empty() {
                    all_full = false;
                }
                patterns.push(joined);
            }
        }
    }
    let combined = regex.compile(&patterns.join("|"), true)?;
    let mut parts = Vec::new();
    let mut last_end = 0usize;
    for matched in regex.find_all(combined, output)? {
        if matched.start > last_end {
            let raw = range_text(output, (last_end, matched.start));
            if !raw.is_empty() {
                let cleaned = strip_hallucinated(
                    regex,
                    raw,
                    &token_patterns,
                    hallucination_patterns,
                    &allowed,
                    removed,
                )?;
                parts.push(ParsedPart::Text(cleaned));
            }
        }
        let mut composition = None;
        for placeholder in &input.placeholders {
            match placeholder {
                InputPlaceholder::Formula(formula) => {
                    let id = regex.compile(&format!("^({})$", formula.pattern), true)?;
                    if !regex.find_all(id, &matched.full)?.is_empty() {
                        composition = Some(ParsedPart::Formula {
                            composition_index: formula.composition_index,
                        });
                        break;
                    }
                }
                InputPlaceholder::Rich(rich) => {
                    let id = regex.compile(&format!("^({})", rich.left_pattern), true)?;
                    if regex.find_all(id, &matched.full)?.is_empty() {
                        continue;
                    }
                    let content_id = regex.compile(
                        &format!("^{}(.*){}$", rich.left_pattern, rich.right_pattern),
                        true,
                    )?;
                    let found = regex.find_all(content_id, &matched.full)?;
                    let raw = found
                        .first()
                        .and_then(|item| item.groups.first())
                        .and_then(|range| *range)
                        .map(|range| range_text(&matched.full, range).to_owned())
                        .unwrap_or_default();
                    let matches_source = rich_text_matches_source(&raw, &rich.source_text);
                    let text = if matches_source {
                        raw
                    } else {
                        strip_hallucinated(
                            regex,
                            &raw,
                            &token_patterns,
                            hallucination_patterns,
                            &allowed,
                            removed,
                        )?
                    };
                    composition = Some(ParsedPart::Rich {
                        composition_index: rich.composition_index,
                        text,
                        matches_source,
                    });
                    break;
                }
            }
        }
        if let Some(composition) = composition {
            parts.push(composition);
        }
        last_end = matched.end;
    }
    if last_end < output.chars().count() {
        let raw = range_text(output, (last_end, output.chars().count()));
        if !raw.is_empty() {
            let cleaned = strip_hallucinated(
                regex,
                raw,
                &token_patterns,
                hallucination_patterns,
                &allowed,
                removed,
            )?;
            parts.push(ParsedPart::Text(cleaned));
        }
    }
    Ok((parts, all_full))
}

/// Render the exact glossary table format used by ILTranslator.
pub fn glossary_block(text: &str, glossaries: &[super::glossary::Glossary]) -> String {
    let active = glossaries
        .iter()
        .filter_map(|glossary| {
            let mut entries = glossary.active_entries_for_text(text);
            entries.sort();
            (!entries.is_empty()).then_some((glossary.name.clone(), entries))
        })
        .collect::<Vec<_>>();
    if active.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        "## Glossary".to_owned(), String::new(),
        "Always use the glossary's **Target Term** for any occurrence of its **Source Term** (including variants, inside tags, or broken across lines).".to_owned(),
        String::new(), "Unlisted terms are translated naturally.".to_owned(), String::new(),
    ];
    for (name, entries) in active {
        lines.push(format!("### Glossary: {name}"));
        lines.push(String::new());
        lines.push("| Source Term | Target Term |\n|-------------|-------------|".to_owned());
        for (source, target) in entries {
            lines.push(format!("| {source} | {target} |"));
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

pub fn role_block(custom: Option<&str>, language: &str) -> String {
    if let Some(custom) = custom.filter(|prompt| !prompt.is_empty()) {
        let mut role = custom.trim().to_owned();
        if !role.contains("Follow all rules strictly.") {
            if !role.ends_with('\n') {
                role.push('\n');
            }
            role.push_str("Follow all rules strictly.");
        }
        role
    } else {
        format!(
            "You are a professional {language} native translator who needs to fluently translate text into {language}.\n\nFollow all rules strictly."
        )
    }
}

pub fn context_block(
    first: Option<(&str, &str)>,
    recent: Option<(&str, &str)>,
    formula_hint: Option<&str>,
) -> String {
    let mut lines = Vec::new();
    if let Some((_, text)) = first {
        lines.push(format!(
            "{}. First title in the full text: {text}",
            lines.len() + 1
        ));
    }
    if let Some((recent_id, text)) = recent {
        if first.is_none_or(|(first_id, _)| first_id != recent_id) {
            lines.push(format!(
                "{}. The most recent title is: {text}",
                lines.len() + 1
            ));
        }
    }
    if let Some(hint) = formula_hint {
        lines.push(format!(
            "{}. Formula placeholder hint:\n{hint}",
            lines.len() + 1
        ));
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!("## Context / Hints\n{}\n", lines.join("\n"))
    }
}

pub const STAGE_NAME: &str = "Translate Paragraphs";
pub const PROMPT_TEMPLATE: &str = "$role_block\n\n## Rules\n\n1. Keep the structure exactly unchanged: do NOT add/remove/reorder any tags, placeholders, or tokens.\n2. Keep all tags unchanged (e.g., <style>, <b>, </style>).\n   - Translate human-readable text inside tags.\n   - Do NOT translate text inside <code>…</code>.\n3. Do NOT translate or alter placeholders: {v1}, {name}, %s, %d, [[...]], %%...%%.\n4. If the entire input is pure code/identifiers, return it unchanged.\n5. Translate ALL human-readable content into $lang_out.\n\n$glossary_block\n\n$context_block\n\n## Output\n\nOutput ONLY the translated $lang_out text. No explanations, no backticks, no extra text.\n\nNow translate the following text:\n\n$text_to_translate";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaceholderRecord {
    Formula {
        id: i64,
        text: String,
        pattern: String,
        chars: Option<String>,
    },
    RichText {
        id: i64,
        left: String,
        right: String,
        left_pattern: String,
        right_pattern: String,
        chars: Option<String>,
    },
}

impl PlaceholderRecord {
    pub fn to_json_value(&self) -> PythonJsonValue {
        let mut fields = vec![(
            "type".into(),
            PythonJsonValue::String(
                match self {
                    Self::Formula { .. } => "formula",
                    Self::RichText { .. } => "rich_text",
                }
                .into(),
            ),
        )];
        match self {
            Self::Formula {
                id,
                text,
                pattern,
                chars,
            } => {
                fields.extend([
                    ("id".into(), PythonJsonValue::Integer(id.to_string())),
                    ("placeholder".into(), PythonJsonValue::String(text.clone())),
                    (
                        "regex_pattern".into(),
                        PythonJsonValue::String(pattern.clone()),
                    ),
                    (
                        "formula_chars".into(),
                        chars
                            .clone()
                            .map_or(PythonJsonValue::Null, PythonJsonValue::String),
                    ),
                ]);
            }
            Self::RichText {
                id,
                left,
                right,
                left_pattern,
                right_pattern,
                chars,
            } => {
                fields.extend([
                    ("id".into(), PythonJsonValue::Integer(id.to_string())),
                    (
                        "left_placeholder".into(),
                        PythonJsonValue::String(left.clone()),
                    ),
                    (
                        "right_placeholder".into(),
                        PythonJsonValue::String(right.clone()),
                    ),
                    (
                        "left_regex_pattern".into(),
                        PythonJsonValue::String(left_pattern.clone()),
                    ),
                    (
                        "right_regex_pattern".into(),
                        PythonJsonValue::String(right_pattern.clone()),
                    ),
                    (
                        "composition_chars".into(),
                        chars
                            .clone()
                            .map_or(PythonJsonValue::Null, PythonJsonValue::String),
                    ),
                ]);
            }
        }
        PythonJsonValue::Object(fields)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParagraphTracking {
    pub input: Option<String>,
    pub output: Option<String>,
    pub pdf_unicode: Option<String>,
    pub placeholders: Vec<PlaceholderRecord>,
    pub original_placeholders: Vec<(String, usize)>,
    pub removed_hallucinated_placeholders: Vec<(String, usize)>,
    pub llm_translate_trackers: Vec<LlmTracking>,
    pub multi_paragraph_id: Option<usize>,
    pub multi_paragraph_index: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LlmTracking {
    pub input: String,
    pub output: String,
    pub has_error: bool,
    pub error_message: String,
    pub placeholder_full_match: bool,
    pub fallback_to_translate: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PageTracking {
    pub paragraph: Vec<ParagraphTracking>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranslationTracking {
    pub page: Vec<PageTracking>,
}

impl TranslationTracking {
    pub fn to_json(&self) -> String {
        let pages = self
            .page
            .iter()
            .map(|page| {
                let paragraphs = page
                    .paragraph
                    .iter()
                    .filter_map(|item| {
                        let (Some(input), Some(pdf_unicode)) = (&item.input, &item.pdf_unicode)
                        else {
                            return None;
                        };
                        Some(PythonJsonValue::Object(vec![
                            ("input".into(), PythonJsonValue::String(input.clone())),
                            (
                                "output".into(),
                                item.output
                                    .clone()
                                    .map_or(PythonJsonValue::Null, PythonJsonValue::String),
                            ),
                            (
                                "pdf_unicode".into(),
                                PythonJsonValue::String(pdf_unicode.clone()),
                            ),
                            (
                                "llm_translate_trackers".into(),
                                PythonJsonValue::Array(
                                    item.llm_translate_trackers
                                        .iter()
                                        .map(LlmTracking::to_json_value)
                                        .collect(),
                                ),
                            ),
                            (
                                "placeholders".into(),
                                PythonJsonValue::Array(
                                    item.placeholders
                                        .iter()
                                        .map(PlaceholderRecord::to_json_value)
                                        .collect(),
                                ),
                            ),
                            (
                                "multi_paragraph_id".into(),
                                item.multi_paragraph_id
                                    .map(|value| PythonJsonValue::Integer(value.to_string()))
                                    .unwrap_or(PythonJsonValue::Null),
                            ),
                            (
                                "multi_paragraph_index".into(),
                                item.multi_paragraph_index
                                    .map(|value| PythonJsonValue::Integer(value.to_string()))
                                    .unwrap_or(PythonJsonValue::Null),
                            ),
                            (
                                "original_placeholders".into(),
                                pairs_to_object(&item.original_placeholders),
                            ),
                            (
                                "removed_hallucinated_placeholders".into(),
                                pairs_to_object(&item.removed_hallucinated_placeholders),
                            ),
                        ]))
                    })
                    .collect();
                PythonJsonValue::Object(vec![(
                    "paragraph".into(),
                    PythonJsonValue::Array(paragraphs),
                )])
            })
            .collect();
        json_dumps(&PythonJsonValue::Object(vec![
            ("cross_page".into(), PythonJsonValue::Array(vec![])),
            ("cross_column".into(), PythonJsonValue::Array(vec![])),
            ("page".into(), PythonJsonValue::Array(pages)),
        ]))
    }
}

impl LlmTracking {
    pub(crate) fn to_json_value(&self) -> PythonJsonValue {
        PythonJsonValue::Object(vec![
            ("input".into(), PythonJsonValue::String(self.input.clone())),
            (
                "output".into(),
                PythonJsonValue::String(self.output.clone()),
            ),
            ("has_error".into(), PythonJsonValue::Bool(self.has_error)),
            (
                "error_message".into(),
                PythonJsonValue::String(self.error_message.clone()),
            ),
            (
                "placeholder_full_match".into(),
                PythonJsonValue::Bool(self.placeholder_full_match),
            ),
            (
                "fallback_to_translate".into(),
                PythonJsonValue::Bool(self.fallback_to_translate),
            ),
        ])
    }
}

fn pairs_to_object(values: &[(String, usize)]) -> PythonJsonValue {
    PythonJsonValue::Object(
        values
            .iter()
            .map(|(key, count)| (key.clone(), PythonJsonValue::Integer(count.to_string())))
            .collect(),
    )
}

pub fn request_priority(text: &str) -> i64 {
    1_048_576 - count_tokens(text) as i64
}

pub fn normalize_response_punctuation(text: &str) -> String {
    static RUN: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"[. 。…，]{20,}").expect("literal regex"));
    RUN.replace_all(text, ".").into_owned()
}

pub fn render_prompt(
    role_block: &str,
    lang_out: &str,
    glossary_block: &str,
    context_block: &str,
    text: &str,
) -> String {
    let values = [
        ("role_block", role_block),
        ("lang_out", lang_out),
        ("glossary_block", glossary_block),
        ("context_block", context_block),
        ("text_to_translate", text),
    ];
    let mut output = String::with_capacity(PROMPT_TEMPLATE.len() + text.len());
    let mut rest = PROMPT_TEMPLATE;
    while let Some(index) = rest.find('$') {
        output.push_str(&rest[..index]);
        let after_dollar = &rest[index + 1..];
        let key_len = after_dollar
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            .count();
        if key_len == 0 {
            output.push('$');
            rest = after_dollar;
            continue;
        }
        let key = &after_dollar[..key_len];
        let value = values
            .iter()
            .find_map(|(name, value)| (*name == key).then_some(*value))
            .expect("all prompt variables are known");
        output.push_str(value);
        rest = &after_dollar[key_len..];
    }
    output.push_str(rest);
    output
}

#[cfg(test)]
mod ordinary_translation_tests;
