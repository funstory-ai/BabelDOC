//! Native planning and Python-compatible response interpretation for the
//! legacy LLM-only translation stage.

use super::ordinary_translation::{PageTracking, ParagraphTracking};
use super::python_semantics::{
    PythonJsonValue as V, clean_json_output, json_dumps, levenshtein_distance_at_most,
};
use super::token_count::count_tokens;

pub const STAGE_NAME: &str = "Translate Paragraphs";

const PROMPT_TEMPLATE: &str = "$role_block\n\n## Structure Rules\n1. Keep **the same number of paragraphs as the input**.\n2. Input paragraphs may be **sliced pieces of the same original paragraph**.  \n   → You MUST treat each input paragraph **as an independent, fixed unit**.  \n   → Do NOT merge paragraphs, split paragraphs, or move content between paragraphs.\n3. Inside each paragraph, you may adjust word order for fluency, but:\n   - Do NOT change the meaning.\n   - Do NOT move placeholders, tags, or code outside their paragraph.\n4. Translate ALL human-readable content into $lang_out.\n\n## Do NOT Modify\n- Tags (e.g., <style>, <b>, <code>): keep them exactly the same.  \n  *Translate tag-internal text except code blocks (<code>…</code>)*.\n- Placeholders: `{v1}`, `{name}`, `%s`, `%d`, `[[...]]`, `%%...%%` — keep exactly unchanged.\n- JSON keys or structure.\n\n$glossary_usage_rules_block\n## Output Format\nReturn a JSON array of the same length.  \nFor each item:\n- Keep the same \"id\" and remove other fields like \"input\" and \"layout_label\".\n- Add \"output\" with the translated text only.\n- No extra text, no ```json blocks.\n\n## Style\n- Produce fluent, professional $lang_out.\n- Preserve punctuation unless needed for target language fluency.\n\n### Example\nInput:\n[\n    {\n    \"id\": 0,\n    \"input\": \"{v1}<style id='2'>hello</style>, world!\",\n    \"layout_label\": \"text\"\n    }\n]\nOutput:\n[\n    {\n    \"id\": 0,\n    \"output\": \"{v1}<style id='2'>你好</style>，世界！\"\n    }\n]\n\n$contextual_hints_block\n\n$glossary_tables_block\n\n## Here is the input:\n\n$json_input_str";

pub fn render_batch_prompt(
    role_block: &str,
    lang_out: &str,
    json_input: &str,
    batch_text_for_glossary: &str,
    title: Option<(&str, &str)>,
    local_title: Option<(&str, &str)>,
    glossaries: &[super::glossary::Glossary],
) -> String {
    let mut contextual_lines = Vec::new();
    if let Some((_, text)) = title {
        contextual_lines.push(format!("1. First title in full text: {text}"));
    }
    if let Some((local_id, text)) = local_title {
        if title.is_none_or(|(title_id, _)| title_id != local_id) {
            contextual_lines.push(format!(
                "{}. The most recent title is: {text}",
                contextual_lines.len() + 1
            ));
        }
    }
    let contextual_hints_block = if contextual_lines.is_empty() {
        String::new()
    } else {
        format!(
            "## Contextual Hints for Better Translation\n{}\n",
            contextual_lines.join("\n")
        )
    };
    let mut by_name = Vec::<(String, Vec<(String, String)>)>::new();
    for glossary in glossaries {
        let mut entries = glossary.active_entries_for_text(batch_text_for_glossary);
        entries.sort();
        if !entries.is_empty() {
            by_name.push((glossary.name.clone(), entries));
        }
    }
    let glossary_usage_rules_block = if by_name.is_empty() {
        String::new()
    } else {
        "## Glossary\nIf a glossary is provided:\n- Always use the exact target term.\n- Apply glossary items even inside tags or when broken by hyphens/line breaks.\n- If glossary does NOT include a term, translate it naturally.\n\n".into()
    };
    let mut tables = Vec::new();
    if !by_name.is_empty() {
        tables.extend(["## Glossary Tables".into(), String::new()]);
        for (name, entries) in by_name {
            tables.push(format!("### Glossary: {name}"));
            tables.push(String::new());
            tables.push("| Source Term | Target Term |\n|-------------|-------------|".into());
            for (source, target) in entries {
                tables.push(format!("| {source} | {target} |"));
            }
            tables.push(String::new());
        }
    }
    let glossary_tables_block = tables.join("\n");
    let mut values = std::collections::HashMap::new();
    values.insert("role_block", role_block);
    values.insert("lang_out", lang_out);
    values.insert(
        "glossary_usage_rules_block",
        glossary_usage_rules_block.as_str(),
    );
    values.insert("contextual_hints_block", contextual_hints_block.as_str());
    values.insert("glossary_tables_block", glossary_tables_block.as_str());
    values.insert("json_input_str", json_input);
    super::python_semantics::template_substitute(PROMPT_TEMPLATE, &values)
        .unwrap_or_else(|_| String::new())
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParagraphFacts {
    pub debug_id: Option<String>,
    pub unicode: Option<String>,
    pub layout_label: Option<String>,
    pub cid: bool,
    pub pure_numeric: bool,
    pub placeholder_only: bool,
    pub box_y2: Option<f64>,
    pub title: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ParagraphId {
    pub page: usize,
    pub index: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GroupKind {
    CrossPage,
    CrossColumn,
    Page,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchPlan {
    pub kind: GroupKind,
    pub id: usize,
    pub paragraphs: Vec<ParagraphId>,
    pub token_count: usize,
}

pub fn is_body(label: Option<&str>) -> bool {
    matches!(label, Some("text" | "plain text" | "paragraph_hybrid"))
}

pub fn stage_total(pages: &[Vec<ParagraphFacts>]) -> usize {
    pages
        .iter()
        .flatten()
        .filter(|p| p.debug_id.is_some() && p.unicode.is_some())
        .count()
}

fn eligible(p: &ParagraphFacts, min_text_length: usize, body: bool) -> bool {
    p.debug_id.is_some()
        && p.unicode
            .as_ref()
            .is_some_and(|s| s.chars().count() >= min_text_length)
        && !p.cid
        && (!body || is_body(p.layout_label.as_deref()))
}

/// Produces submissions in source order. `multi_paragraph_id` is assigned at
/// submission time, before any worker can finish.
pub fn plan_batches(pages: &[Vec<ParagraphFacts>], min_text_length: usize) -> Vec<BatchPlan> {
    let mut translated = std::collections::HashSet::<ParagraphId>::new();
    let mut batches = Vec::new();
    let mut next_id = 1usize;
    for page in 0..pages.len().saturating_sub(1) {
        let left = pages[page]
            .iter()
            .enumerate()
            .filter(|(_, p)| eligible(p, min_text_length, true))
            .map(|(index, _)| ParagraphId { page, index })
            .filter(|id| !translated.contains(id))
            .last();
        let right = pages[page + 1]
            .iter()
            .enumerate()
            .find(|(_, p)| eligible(p, min_text_length, true))
            .map(|(index, _)| ParagraphId {
                page: page + 1,
                index,
            })
            .filter(|id| !translated.contains(id));
        if let (Some(left), Some(right)) = (left, right) {
            let token_count = [left, right]
                .iter()
                .map(|id| {
                    pages[id.page][id.index]
                        .unicode
                        .as_deref()
                        .unwrap_or_default()
                })
                .map(count_tokens)
                .sum();
            translated.insert(left);
            translated.insert(right);
            batches.push(BatchPlan {
                kind: GroupKind::CrossPage,
                id: next_id,
                paragraphs: vec![left, right],
                token_count,
            });
            next_id += 1;
        }
    }
    for (page, values) in pages.iter().enumerate() {
        let body = values
            .iter()
            .enumerate()
            .filter(|(_, p)| eligible(p, min_text_length, true))
            .map(|(index, _)| ParagraphId { page, index })
            .filter(|id| !translated.contains(id))
            .collect::<Vec<_>>();
        for pair in body.windows(2) {
            let first = pair[0];
            let second = pair[1];
            if translated.contains(&first) || translated.contains(&second) {
                continue;
            }
            let y1 = pages[first.page][first.index].box_y2;
            let y2 = pages[second.page][second.index].box_y2;
            if !matches!((y1, y2), (Some(a), Some(b)) if b - a > 20.0) {
                continue;
            }
            let token_count = [first, second]
                .iter()
                .map(|id| {
                    pages[id.page][id.index]
                        .unicode
                        .as_deref()
                        .unwrap_or_default()
                })
                .map(count_tokens)
                .sum();
            translated.insert(first);
            translated.insert(second);
            batches.push(BatchPlan {
                kind: GroupKind::CrossColumn,
                id: next_id,
                paragraphs: vec![first, second],
                token_count,
            });
            next_id += 1;
        }
    }
    for (page, values) in pages.iter().enumerate() {
        let mut group = Vec::new();
        let mut tokens = 0usize;
        for (index, paragraph) in values.iter().enumerate() {
            let id = ParagraphId { page, index };
            if translated.contains(&id) {
                continue;
            }
            if paragraph.debug_id.is_none() || paragraph.unicode.is_none() {
                continue;
            }
            if paragraph.cid {
                continue;
            }
            let text = paragraph.unicode.as_deref().unwrap_or_default();
            if text.chars().count() < min_text_length
                || paragraph.pure_numeric
                || paragraph.placeholder_only
            {
                continue;
            }
            tokens += count_tokens(text);
            group.push(id);
            if paragraph.title {
                // Recent-title selection is maintained by the bridge while scanning.
            }
            if tokens > 200 || group.len() > 5 {
                batches.push(BatchPlan {
                    kind: GroupKind::Page,
                    id: next_id,
                    paragraphs: std::mem::take(&mut group),
                    token_count: tokens,
                });
                next_id += 1;
                tokens = 0;
            }
        }
        if !group.is_empty() {
            batches.push(BatchPlan {
                kind: GroupKind::Page,
                id: next_id,
                paragraphs: group,
                token_count: tokens,
            });
            next_id += 1;
        }
    }
    batches
}

pub fn batch_json(
    values: &[(String, Option<String>, Option<String>)],
    formula_hints: bool,
) -> String {
    let items = values
        .iter()
        .enumerate()
        .map(|(id, (input, label, hint))| {
            let mut fields = vec![
                ("id".into(), V::Integer(id.to_string())),
                ("input".into(), V::String(input.clone())),
                (
                    "layout_label".into(),
                    label.clone().map_or(V::Null, V::String),
                ),
            ];
            if formula_hints && let Some(hint) = hint {
                fields.push(("formula_placeholders_hint".into(), V::String(hint.clone())));
            }
            V::Object(fields)
        })
        .collect();
    json_dumps(&V::Array(items))
}

#[derive(Clone, Debug, PartialEq)]
pub struct InterpretedItem {
    pub id: usize,
    pub output: String,
    pub input_index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InterpretResult {
    pub items: Vec<InterpretedItem>,
    pub fallback_indices: Vec<usize>,
    pub fallback_errors: Vec<(usize, String, bool)>,
    pub processed: usize,
    pub ok_count: usize,
    pub fallback_count: usize,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DocumentTracking {
    pub cross_page: Vec<PageTracking>,
    pub cross_column: Vec<PageTracking>,
    pub page: Vec<PageTracking>,
}

impl DocumentTracking {
    pub fn new(page_lengths: &[usize]) -> Self {
        Self {
            cross_page: Vec::new(),
            cross_column: Vec::new(),
            page: page_lengths
                .iter()
                .map(|length| PageTracking {
                    paragraph: vec![ParagraphTracking::default(); *length],
                })
                .collect(),
        }
    }

    pub fn new_group(&mut self, kind: GroupKind, paragraph_count: usize) -> &mut PageTracking {
        let values = match kind {
            GroupKind::CrossPage => &mut self.cross_page,
            GroupKind::CrossColumn => &mut self.cross_column,
            GroupKind::Page => &mut self.page,
        };
        values.push(PageTracking {
            paragraph: vec![ParagraphTracking::default(); paragraph_count],
        });
        values.last_mut().expect("just pushed tracking group")
    }

    pub fn to_json(&self) -> String {
        use super::ordinary_translation::LlmTracking;
        use super::python_semantics::{PythonJsonValue as V, json_dumps};
        fn convert(page: &PageTracking) -> V {
            let paragraphs = page
                .paragraph
                .iter()
                .filter_map(|item| {
                    let (Some(input), Some(pdf_unicode)) = (&item.input, &item.pdf_unicode) else {
                        return None;
                    };
                    Some(V::Object(vec![
                        ("input".into(), V::String(input.clone())),
                        (
                            "output".into(),
                            item.output.clone().map_or(V::Null, V::String),
                        ),
                        ("pdf_unicode".into(), V::String(pdf_unicode.clone())),
                        (
                            "llm_translate_trackers".into(),
                            V::Array(
                                item.llm_translate_trackers
                                    .iter()
                                    .map(LlmTracking::to_json_value)
                                    .collect(),
                            ),
                        ),
                        (
                            "placeholders".into(),
                            V::Array(
                                item.placeholders
                                    .iter()
                                    .map(|v| v.to_json_value())
                                    .collect(),
                            ),
                        ),
                        (
                            "multi_paragraph_id".into(),
                            item.multi_paragraph_id
                                .map(|v| V::Integer(v.to_string()))
                                .unwrap_or(V::Null),
                        ),
                        (
                            "multi_paragraph_index".into(),
                            item.multi_paragraph_index
                                .map(|v| V::Integer(v.to_string()))
                                .unwrap_or(V::Null),
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
            V::Object(vec![("paragraph".into(), V::Array(paragraphs))])
        }
        fn pairs_to_object(values: &[(String, usize)]) -> V {
            V::Object(
                values
                    .iter()
                    .map(|(key, value)| (key.clone(), V::Integer(value.to_string())))
                    .collect(),
            )
        }
        json_dumps(&V::Object(vec![
            (
                "cross_page".into(),
                V::Array(self.cross_page.iter().map(convert).collect()),
            ),
            (
                "cross_column".into(),
                V::Array(self.cross_column.iter().map(convert).collect()),
            ),
            (
                "page".into(),
                V::Array(self.page.iter().map(convert).collect()),
            ),
        ]))
    }
}

#[derive(Clone, Debug, PartialEq)]
enum CompatKey {
    Null,
    Bool(bool),
    Integer(String),
    Float(f64),
    String(String),
    Nan(usize),
}

fn compat_key(value: &V, nan_sequence: usize) -> CompatKey {
    match value {
        V::Null => CompatKey::Null,
        V::Bool(value) => CompatKey::Bool(*value),
        V::Integer(value) => CompatKey::Integer(value.clone()),
        V::Float(value) if value.is_nan() => CompatKey::Nan(nan_sequence),
        V::Float(value) => CompatKey::Float(*value),
        V::String(value) => CompatKey::String(value.clone()),
        _ => CompatKey::String(format!("container:{nan_sequence}")),
    }
}

fn keys_equal(left: &CompatKey, right: &CompatKey) -> bool {
    fn bool_int(value: bool) -> &'static str {
        if value { "1" } else { "0" }
    }
    fn int_float(integer: &str, float: f64) -> bool {
        float.is_finite() && float.fract() == 0.0 && format!("{float:.0}") == integer
    }
    match (left, right) {
        (CompatKey::Null, CompatKey::Null) => true,
        (CompatKey::Bool(a), CompatKey::Bool(b)) => a == b,
        (CompatKey::Integer(a), CompatKey::Integer(b)) => a == b,
        (CompatKey::Float(a), CompatKey::Float(b)) => a == b,
        (CompatKey::Bool(a), CompatKey::Integer(b))
        | (CompatKey::Integer(b), CompatKey::Bool(a)) => bool_int(*a) == b,
        (CompatKey::Bool(a), CompatKey::Float(b)) | (CompatKey::Float(b), CompatKey::Bool(a)) => {
            int_float(bool_int(*a), *b)
        }
        (CompatKey::Integer(a), CompatKey::Float(b))
        | (CompatKey::Float(b), CompatKey::Integer(a)) => int_float(a, *b),
        (CompatKey::String(a), CompatKey::String(b)) => a == b,
        (CompatKey::Nan(a), CompatKey::Nan(b)) => a == b,
        _ => false,
    }
}

fn python_int(value: &V) -> Result<i64, String> {
    match value {
        V::Integer(value) => value.parse().map_err(|_| "int too large".into()),
        V::Bool(value) => Ok(i64::from(*value)),
        V::Float(value) if value.is_finite() => Ok(*value as i64),
        V::String(value) => value
            .trim()
            .parse::<i64>()
            .map_err(|_| format!("invalid literal for int() with base 10: '{value}'")),
        _ => Err(format!(
            "int() argument must be a string, a bytes-like object or a real number, not {}",
            type_name(value)
        )),
    }
}

fn type_name(value: &V) -> &'static str {
    match value {
        V::Null => "NoneType",
        V::Bool(_) => "bool",
        V::Integer(_) => "int",
        V::Float(_) => "float",
        V::String(_) => "str",
        V::Array(_) => "list",
        V::Object(_) => "dict",
    }
}

fn output_text(value: &V) -> Option<String> {
    match value {
        V::String(value) => Some(value.clone()),
        _ => None,
    }
}

/// Mirrors the dict-comprehension followed by the per-result try/except/finally
/// structure. The caller applies the returned texts and queues fallback jobs.
pub fn interpret_response(
    response: &str,
    inputs: &[String],
    disable_same_text_fallback: bool,
) -> InterpretResult {
    let parsed = match super::python_semantics::json_loads(clean_json_output(response).trim()) {
        Ok(value) => value,
        Err(error) => {
            return InterpretResult {
                items: vec![],
                fallback_indices: (0..inputs.len()).collect(),
                fallback_errors: vec![],
                processed: 0,
                ok_count: 0,
                fallback_count: 0,
                error: Some(python_json_error(response, &error.to_string())),
            };
        }
    };
    let parsed = match parsed {
        V::Object(fields)
            if fields
                .iter()
                .rfind(|(k, _)| k == "output")
                .or_else(|| fields.iter().rfind(|(k, _)| k == "input"))
                .is_some_and(|(_, value)| truthy(value)) =>
        {
            V::Array(vec![V::Object(fields)])
        }
        value => value,
    };
    let values = match parsed {
        V::Array(values) => values,
        _ => {
            return InterpretResult {
                items: vec![],
                fallback_indices: (0..inputs.len()).collect(),
                fallback_errors: vec![],
                processed: 0,
                ok_count: 0,
                fallback_count: 0,
                error: Some("'{}' object is not iterable".into()),
            };
        }
    };
    let mut results = Vec::<(CompatKey, V)>::new();
    let mut nan_sequence = 0usize;
    for item in values {
        let V::Object(fields) = item else {
            return InterpretResult {
                items: vec![],
                fallback_indices: (0..inputs.len()).collect(),
                fallback_errors: vec![],
                processed: 0,
                ok_count: 0,
                fallback_count: 0,
                error: Some("list indices must be integers or slices, not str".into()),
            };
        };
        let Some((_, id)) = fields.iter().rfind(|(key, _)| key == "id") else {
            return InterpretResult {
                items: vec![],
                fallback_indices: (0..inputs.len()).collect(),
                fallback_errors: vec![],
                processed: 0,
                ok_count: 0,
                fallback_count: 0,
                error: Some("'id'".into()),
            };
        };
        let output = fields
            .iter()
            .rfind(|(key, _)| key == "output")
            .map(|(_, v)| v)
            .or_else(|| {
                fields
                    .iter()
                    .rfind(|(key, _)| key == "input")
                    .map(|(_, v)| v)
            })
            .cloned()
            .unwrap_or(V::Null);
        let key = compat_key(id, nan_sequence);
        if matches!(key, CompatKey::Nan(_)) {
            nan_sequence += 1;
        }
        if let Some(existing) = results
            .iter_mut()
            .find(|(candidate, _)| keys_equal(candidate, &key))
        {
            existing.1 = output;
        } else {
            results.push((key, output));
        }
    }
    if results.len() != inputs.len() {
        return InterpretResult {
            items: vec![],
            fallback_indices: (0..inputs.len()).collect(),
            fallback_errors: vec![],
            processed: 0,
            ok_count: 0,
            fallback_count: 0,
            error: Some(format!(
                "Translation results length mismatch. Expected: {}, Got: {}",
                inputs.len(),
                results.len()
            )),
        };
    }
    let mut items = Vec::new();
    let mut fallback_errors = Vec::new();
    let mut fallback_indices = Vec::new();
    let mut processed = 0;
    let mut ok_count = 0;
    let mut fallback_count = 0;
    for (key, output) in results {
        let raw = key_to_value(&key);
        processed += 1;
        let id = match python_int(&raw) {
            Ok(id) => id,
            Err(error) => {
                fallback_count += 1;
                return InterpretResult {
                    items: vec![],
                    fallback_indices: (0..inputs.len()).collect(),
                    fallback_errors: vec![],
                    processed,
                    ok_count,
                    fallback_count,
                    error: Some(match raw {
                        V::String(_) => "list indices must be integers or slices, not str".into(),
                        V::Float(_) => "list indices must be integers or slices, not float".into(),
                        V::Null => "list indices must be integers or slices, not NoneType".into(),
                        V::Object(_) => "list indices must be integers or slices, not dict".into(),
                        V::Array(_) => "list indices must be integers or slices, not list".into(),
                        _ => error,
                    }),
                };
            }
        };
        if id >= inputs.len() as i64 || id < -(inputs.len() as i64) {
            return InterpretResult {
                items: vec![],
                fallback_indices: (0..inputs.len()).collect(),
                fallback_errors: vec![],
                processed,
                ok_count,
                fallback_count: fallback_count + 1,
                error: Some("list index out of range".into()),
            };
        }
        let index = if id < 0 { inputs.len() as i64 + id } else { id } as usize;
        if output_text(&output).is_none() {
            fallback_count += 1;
            fallback_indices.push(index);
            continue;
        }
        let translated = output_text(&output).unwrap();
        let translated = collapse_punctuation(&translated);
        let source = collapse_punctuation(&inputs[index]);
        let in_tokens = count_tokens(&source);
        let out_tokens = count_tokens(&translated);
        let same_as_input = !disable_same_text_fallback && source == translated && in_tokens > 10;
        let too_long_or_short = in_tokens == 0
            || !(0.3 < out_tokens as f64 / in_tokens as f64
                && (out_tokens as f64 / in_tokens as f64) < 3.0);
        let edit_distance = if !disable_same_text_fallback
            && !same_as_input
            && !too_long_or_short
            && in_tokens > 20
        {
            levenshtein_distance_at_most(&inputs[index], &translated, 4)
        } else {
            None
        };
        let too_similar = edit_distance.is_some();
        let valid = !same_as_input && !too_long_or_short && !too_similar;
        if valid {
            ok_count += 1;
            items.push(InterpretedItem {
                id: index,
                output: translated,
                input_index: index,
            });
        } else if same_as_input {
            fallback_count += 1;
            fallback_indices.push(index);
            fallback_errors.push((
                index,
                "Translation result is the same as input, fallback.".into(),
                true,
            ));
        } else if too_long_or_short {
            fallback_count += 1;
            fallback_indices.push(index);
            fallback_errors.push((index, format!("Translation result is too long or too short. Input: {in_tokens}, Output: {out_tokens}"), true));
        } else if let Some(distance) = edit_distance {
            fallback_count += 1;
            fallback_indices.push(index);
            fallback_errors.push((index, format!("Translation result edit distance is too small. distance: {distance}, input: {}, output: {translated}", inputs[index]), true));
        }
    }
    InterpretResult {
        items,
        fallback_indices,
        fallback_errors,
        processed,
        ok_count,
        fallback_count,
        error: None,
    }
}

fn python_json_error(response: &str, parser_error: &str) -> String {
    if parser_error.starts_with("invalid literal at byte") {
        let column = response
            .chars()
            .take_while(|character| character.is_whitespace())
            .count()
            + 1;
        let offset = column - 1;
        return format!("Expecting value: line 1 column {column} (char {offset})");
    }
    parser_error.to_owned()
}

fn key_to_value(key: &CompatKey) -> V {
    match key {
        CompatKey::Null => V::Null,
        CompatKey::Bool(value) => V::Bool(*value),
        CompatKey::Integer(value) => V::Integer(value.clone()),
        CompatKey::Float(value) => V::Float(*value),
        CompatKey::String(value) => V::String(value.clone()),
        CompatKey::Nan(_) => V::Float(f64::NAN),
    }
}

fn truthy(value: &V) -> bool {
    match value {
        V::Null => false,
        V::Bool(v) => *v,
        V::Integer(v) => v != "0",
        V::Float(v) => *v != 0.0,
        V::String(v) => !v.is_empty(),
        V::Array(v) => !v.is_empty(),
        V::Object(v) => !v.is_empty(),
    }
}

fn collapse_punctuation(value: &str) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut index = 0;
    while index < chars.len() {
        if matches!(chars[index], '.' | ' ' | '。' | '…' | '，') {
            let start = index;
            while index < chars.len() && matches!(chars[index], '.' | ' ' | '。' | '…' | '，') {
                index += 1;
            }
            if index - start >= 20 {
                out.push('.');
            } else {
                out.extend(&chars[start..index]);
            }
        } else {
            out.push(chars[index]);
            index += 1;
        }
    }
    out
}

#[cfg(test)]
mod llm_only_tests;
