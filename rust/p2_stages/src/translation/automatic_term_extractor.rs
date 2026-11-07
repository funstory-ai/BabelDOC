use crate::translation::{
    glossary::Glossary,
    python_semantics::{
        PythonJsonValue, clean_json_output, format_named_template, json_dumps, json_loads,
    },
    token_count::count_tokens,
};

pub const STAGE_NAME: &str = "Automatic Term Extraction";
pub const LLM_PROMPT_TEMPLATE: &str = r#"
You are an expert multilingual terminologist. Extract key terms from the text and translate them into {target_language}.

### Extraction Rules
1. Include only: named entities (people, orgs, locations, theorem/algorithm names, dates) and domain-specific nouns/noun phrases essential to meaning.
2. No full sentences. Ignore function words.
3. Use minimal noun phrases (≤5 words unless a named entity). No generic academic nouns (e.g., model, case, property) unless part of a standard term.
4. No mathematical items: variables (X1, a, ε), symbols (=, +, →, ⊥⊥, ∈), subscripts/superscripts, formula fragments, mappings (T: H1→H2), etc. Keep only natural-language concepts.
5. Extract each term once. Keep order of first appearance.

### Translation Rules
1. Translate each term into {target_language}.
2. If in the reference glossary, use its translation exactly.
3. Keep proper names in original language unless a well-known translation exists.
4. Ensure consistent translations.

{reference_glossary_section}

### Output Format
- Return ONLY a valid JSON array.
- Each element: {{"src": "...", "tgt": "..."}}.
- No comments, no backticks, no extra text.
- If no terms: [].

### Example
For terms “LLM”, “GPT”:
{example_output}

Input Text:
```
{text_to_process}
```

Return JSON ONLY. NO OTHER TEXT.
Result:
"#;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TermParagraphFacts {
    pub debug_id: Option<String>,
    pub unicode: Option<String>,
    pub cid: bool,
    pub pure_numeric: bool,
    pub placeholder_only: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrackingParagraph {
    pub pdf_unicodes: Vec<String>,
    pub output: Option<PythonJsonValue>,
    pub input: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrackingPage {
    pub paragraph: Vec<TrackingParagraph>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TermTracking {
    pub page: Vec<TrackingPage>,
}

#[derive(Clone, Debug)]
pub struct TermBatch {
    pub page_index: usize,
    pub tracking_index: usize,
    pub paragraphs: Vec<String>,
    pub token_count: usize,
    pub prompt: String,
}

#[derive(Clone, Debug)]
pub enum PageAction {
    Skipped,
    Batch(TermBatch),
}

impl TermTracking {
    pub fn to_json(&self) -> String {
        let pages = self
            .page
            .iter()
            .map(|page| {
                let paragraphs = page
                    .paragraph
                    .iter()
                    .filter(|paragraph| !paragraph.pdf_unicodes.is_empty())
                    .map(|paragraph| {
                        PythonJsonValue::Object(vec![
                            (
                                "pdf_unicodes".into(),
                                PythonJsonValue::Array(
                                    paragraph
                                        .pdf_unicodes
                                        .iter()
                                        .cloned()
                                        .map(PythonJsonValue::String)
                                        .collect(),
                                ),
                            ),
                            (
                                "output".into(),
                                paragraph.output.clone().unwrap_or(PythonJsonValue::Null),
                            ),
                            (
                                "input".into(),
                                paragraph
                                    .input
                                    .clone()
                                    .map(PythonJsonValue::String)
                                    .unwrap_or(PythonJsonValue::Null),
                            ),
                        ])
                    })
                    .collect();
                PythonJsonValue::Object(vec![(
                    "paragraph".into(),
                    PythonJsonValue::Array(paragraphs),
                )])
            })
            .collect();
        json_dumps(&PythonJsonValue::Object(vec![(
            "page".into(),
            PythonJsonValue::Array(pages),
        )]))
    }
}

pub fn plan_page(
    page_index: usize,
    facts: &[TermParagraphFacts],
    lang_out: &str,
    user_glossaries: &[Glossary],
    tracking: &mut TermTracking,
) -> (Vec<TermBatch>, usize) {
    let actions = plan_page_actions(page_index, facts, lang_out, user_glossaries, tracking);
    actions
        .into_iter()
        .fold((Vec::new(), 0), |mut result, action| {
            match action {
                PageAction::Skipped => result.1 += 1,
                PageAction::Batch(batch) => result.0.push(batch),
            }
            result
        })
}

pub fn plan_page_actions(
    page_index: usize,
    facts: &[TermParagraphFacts],
    lang_out: &str,
    user_glossaries: &[Glossary],
    tracking: &mut TermTracking,
) -> Vec<PageAction> {
    let mut actions = Vec::new();
    let mut paragraphs = Vec::new();
    let mut tokens = 0;
    let page_tracking = &mut tracking.page[page_index];
    for facts in facts {
        let Some(text) = facts.unicode.as_deref() else {
            actions.push(PageAction::Skipped);
            continue;
        };
        if facts.debug_id.is_none() || facts.cid || facts.pure_numeric || facts.placeholder_only {
            actions.push(PageAction::Skipped);
            continue;
        }
        tokens += count_tokens(text);
        paragraphs.push(text.to_owned());
        if tokens > 600 || paragraphs.len() > 12 {
            actions.push(PageAction::Batch(make_batch(
                page_index,
                paragraphs,
                tokens,
                lang_out,
                user_glossaries,
                page_tracking,
            )));
            paragraphs = Vec::new();
            tokens = 0;
        }
    }
    if !paragraphs.is_empty() {
        actions.push(PageAction::Batch(make_batch(
            page_index,
            paragraphs,
            tokens,
            lang_out,
            user_glossaries,
            page_tracking,
        )));
    }
    actions
}

fn make_batch(
    page_index: usize,
    paragraphs: Vec<String>,
    token_count: usize,
    lang_out: &str,
    user_glossaries: &[Glossary],
    tracking_page: &mut TrackingPage,
) -> TermBatch {
    let text = paragraphs.join("\n\n");
    let reference = reference_glossary_section(user_glossaries, &text);
    let values = std::collections::HashMap::from([
        ("target_language", lang_out),
        ("reference_glossary_section", reference.as_str()),
        (
            "example_output",
            "[\n  {\"src\": \"LLM\", \"tgt\": \"大语言模型\"},\n  {\"src\": \"GPT\", \"tgt\": \"GPT\"}\n]",
        ),
        ("text_to_process", text.as_str()),
    ]);
    let prompt = format_named_template(LLM_PROMPT_TEMPLATE, &values)
        .expect("fixed automatic term extraction prompt template is valid");
    let tracking_index = tracking_page.paragraph.len();
    tracking_page.paragraph.push(TrackingParagraph {
        pdf_unicodes: Vec::new(),
        output: None,
        input: None,
    });
    TermBatch {
        page_index,
        tracking_index,
        paragraphs,
        token_count,
        prompt,
    }
}

pub fn record_input(tracking: &mut TermTracking, batch: &TermBatch) {
    let paragraph = &mut tracking.page[batch.page_index].paragraph[batch.tracking_index];
    paragraph.pdf_unicodes.clone_from(&batch.paragraphs);
    paragraph.input = Some(batch.prompt.clone());
}

fn reference_glossary_section(glossaries: &[Glossary], text: &str) -> String {
    let mut grouped: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for glossary in glossaries {
        let active = glossary.active_entries_for_text(text);
        if !active.is_empty() {
            if let Some((_, entries)) = grouped.iter_mut().find(|(name, _)| name == &glossary.name)
            {
                *entries = active;
            } else {
                grouped.push((glossary.name.clone(), active));
            }
        }
    }
    if grouped.is_empty() {
        return String::new();
    }
    let mut result = String::from("Reference Glossaries (for consistency and quality):\n");
    for (name, entries) in grouped {
        result.push_str(&format!("\n{name}:\n"));
        let mut entries = entries;
        entries.sort();
        entries.dedup();
        for (source, target) in entries {
            result.push_str(&format!("- {source} → {target}\n"));
        }
    }
    result.push_str("\nPlease consider these existing translations for consistency when extracting new terms. IMPORTANT: You should also extract terms that appear in the reference glossaries above if they are found in the input text - don't skip them just because they already exist in the reference.");
    result
}

pub fn parse_terms(response: &str) -> Result<Vec<(String, String)>, String> {
    let cleaned = clean_json_output(response);
    let value = json_loads(&cleaned).map_err(|error| error.to_string())?;
    let values = match value {
        PythonJsonValue::Array(values) => values,
        value => vec![value],
    };
    let mut terms = Vec::new();
    for value in values {
        let PythonJsonValue::Object(fields) = value else {
            continue;
        };
        let Some(source) = fields
            .iter()
            .find(|(key, _)| key == "src")
            .map(|(_, value)| value)
        else {
            continue;
        };
        let Some(target) = fields
            .iter()
            .find(|(key, _)| key == "tgt")
            .map(|(_, value)| value)
        else {
            continue;
        };
        let source = python_str(source).trim().to_owned();
        let target = python_str(target).trim().to_owned();
        if source == target && source.chars().count() < 3 {
            continue;
        }
        if !source.is_empty() && !target.is_empty() && source.chars().count() < 100 {
            terms.push((source, target));
        }
    }
    Ok(terms)
}

fn python_str(value: &PythonJsonValue) -> String {
    match value {
        PythonJsonValue::Null => "None".into(),
        PythonJsonValue::Bool(value) => if *value { "True" } else { "False" }.into(),
        PythonJsonValue::Integer(value) => value.clone(),
        PythonJsonValue::Float(value) => super::python_semantics::python_float_str(*value),
        PythonJsonValue::String(value) => value.clone(),
        PythonJsonValue::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        PythonJsonValue::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    python_repr(&PythonJsonValue::String(key.clone())),
                    python_repr(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_repr(value: &PythonJsonValue) -> String {
    match value {
        PythonJsonValue::String(text) => {
            format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'"))
        }
        PythonJsonValue::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        PythonJsonValue::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    python_repr(&PythonJsonValue::String(key.clone())),
                    python_repr(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => python_str(other),
    }
}

pub fn record_output(tracking: &mut TermTracking, batch: &TermBatch, output: PythonJsonValue) {
    tracking.page[batch.page_index].paragraph[batch.tracking_index].output = Some(output);
}

pub fn frequency_json(raw_terms: &[(String, String)]) -> String {
    let values = raw_terms
        .iter()
        .map(|(source, target)| {
            PythonJsonValue::Array(vec![
                PythonJsonValue::String(source.clone()),
                PythonJsonValue::String(target.clone()),
            ])
        })
        .collect();
    json_dumps(&PythonJsonValue::Array(values))
}

pub fn batch_priority(token_count: usize) -> i64 {
    1_048_576 - token_count as i64
}

#[cfg(test)]
mod automatic_term_extractor_tests;
