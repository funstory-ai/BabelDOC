//! Per-document state shared by all split parts.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::translation::glossary::{Glossary, finalize_terms, normalize_source};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TitleContextSnapshot {
    pub debug_id: Option<String>,
    pub unicode: Option<String>,
    pub layout_label: Option<String>,
}

#[derive(Default)]
struct State {
    first_paragraph: Option<TitleContextSnapshot>,
    recent_title_paragraph: Option<TitleContextSnapshot>,
    user_glossaries: Vec<Glossary>,
    auto_extracted_glossary: Option<Glossary>,
    raw_extracted_terms: Vec<(String, String)>,
    auto_enabled_ocr_workaround: bool,
    valid_char_count_total: i64,
    total_valid_text_token_count: i64,
    unique_name: String,
    norm_terms: HashSet<String>,
}

/// Cloneable handle to the one mutable context owned by a document run.
#[derive(Clone, Default)]
pub struct SharedContextRoot(Arc<Mutex<State>>);

impl SharedContextRoot {
    pub fn new(user_glossaries: Vec<Glossary>) -> Self {
        let root = Self::default();
        root.initialize_glossaries(user_glossaries);
        root
    }

    pub fn initialize_glossaries(&self, user_glossaries: Vec<Glossary>) {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.user_glossaries = user_glossaries;
        state.auto_extracted_glossary = None;
        state.raw_extracted_terms.clear();
        state.unique_name = unique_auto_name(&state.user_glossaries);
        state.norm_terms = state
            .user_glossaries
            .iter()
            .flat_map(|glossary| glossary.entries.iter())
            .map(|entry| normalize_source(&entry.source))
            .collect();
        state.valid_char_count_total = 0;
        state.total_valid_text_token_count = 0;
    }

    pub fn set_auto_enabled_ocr_workaround(&self, value: bool) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .auto_enabled_ocr_workaround = value;
    }

    pub fn auto_enabled_ocr_workaround(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .auto_enabled_ocr_workaround
    }

    pub fn add_raw_extracted_term_pair(&self, source: String, target: String) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .raw_extracted_terms
            .push((source, target));
    }

    pub fn raw_extracted_terms(&self) -> Vec<(String, String)> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .raw_extracted_terms
            .clone()
    }

    pub fn unique_name(&self) -> String {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .unique_name
            .clone()
    }

    pub fn contains_term(&self, term: &str) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .norm_terms
            .contains(&normalize_source(term))
    }

    pub fn finalize_auto_extracted_glossary(&self) -> Result<(), String> {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.auto_extracted_glossary =
            finalize_terms(state.unique_name.clone(), &state.raw_extracted_terms)
                .map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn user_glossaries(&self) -> Vec<Glossary> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .user_glossaries
            .clone()
    }

    pub fn auto_extracted_glossary(&self) -> Option<Glossary> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .auto_extracted_glossary
            .clone()
    }

    pub fn get_glossaries(&self) -> Vec<Glossary> {
        let state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut result = state.user_glossaries.clone();
        if let Some(glossary) = &state.auto_extracted_glossary {
            result.push(glossary.clone());
        }
        result
    }

    pub fn get_glossaries_for_translation(&self, auto_extract_enabled: bool) -> Vec<Glossary> {
        let state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if auto_extract_enabled && let Some(glossary) = &state.auto_extracted_glossary {
            return vec![glossary.clone()];
        }
        let mut result = state.user_glossaries.clone();
        if let Some(glossary) = &state.auto_extracted_glossary {
            result.push(glossary.clone());
        }
        result
    }

    pub fn add_valid_counts(&self, char_count: i64, token_count: i64) {
        if char_count <= 0 && token_count <= 0 {
            return;
        }
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if char_count > 0 {
            state.valid_char_count_total += char_count;
        }
        if token_count > 0 {
            state.total_valid_text_token_count += token_count;
        }
    }

    pub fn valid_counts(&self) -> (i64, i64) {
        let state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (
            state.valid_char_count_total,
            state.total_valid_text_token_count,
        )
    }

    pub fn first_paragraph(&self) -> Option<TitleContextSnapshot> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .first_paragraph
            .clone()
    }
    pub fn set_first_paragraph(&self, value: Option<TitleContextSnapshot>) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .first_paragraph = value;
    }
    pub fn recent_title_paragraph(&self) -> Option<TitleContextSnapshot> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .recent_title_paragraph
            .clone()
    }
    pub fn set_recent_title_paragraph(&self, value: Option<TitleContextSnapshot>) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .recent_title_paragraph = value;
    }
}

fn unique_auto_name(glossaries: &[Glossary]) -> String {
    let names: HashSet<&str> = glossaries
        .iter()
        .map(|glossary| glossary.name.as_str())
        .collect();
    let base = "auto_extracted_glossary";
    if !names.contains(base) {
        return base.to_owned();
    }
    let mut suffix = 1;
    loop {
        let candidate = format!("{base}#{suffix}");
        if !names.contains(candidate.as_str()) {
            return candidate;
        }
        suffix += 1;
    }
}

#[cfg(test)]
mod shared_context_tests;
