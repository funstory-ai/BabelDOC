use std::collections::HashMap;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

use crate::shared::layout_helper::unicode::is_python_strip_whitespace;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlossaryEntry {
    pub source: String,
    pub target: String,
    pub target_language: Option<String>,
}

impl GlossaryEntry {
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            target_language: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Glossary {
    pub name: String,
    pub entries: Vec<GlossaryEntry>,
    matcher: Option<AhoCorasick>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlossaryError {
    EmptyPattern,
    Matcher(String),
}

impl std::fmt::Display for GlossaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPattern => f.write_str("empty glossary source is not supported"),
            Self::Matcher(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for GlossaryError {}

impl Glossary {
    pub fn new(
        name: impl Into<String>,
        entries: Vec<GlossaryEntry>,
    ) -> Result<Self, GlossaryError> {
        Self::new_with_source_spelling_votes(name, entries, false)
    }

    fn new_with_source_spelling_votes(
        name: impl Into<String>,
        entries: Vec<GlossaryEntry>,
        prefer_frequent_spelling: bool,
    ) -> Result<Self, GlossaryError> {
        let mut groups: Vec<Vec<(GlossaryEntry, usize)>> = Vec::new();
        let mut group_indices: HashMap<String, usize> = HashMap::new();
        for entry in entries {
            let normalized = normalize_source(&entry.source);
            if let Some(index) = group_indices.get(&normalized).copied() {
                if !prefer_frequent_spelling {
                    continue;
                }
                let group = &mut groups[index];
                if let Some((_, count)) = group
                    .iter_mut()
                    .find(|(candidate, _)| candidate.source == entry.source)
                {
                    *count += 1;
                } else {
                    group.push((entry, 1));
                }
            } else {
                group_indices.insert(normalized, groups.len());
                groups.push(vec![(entry, 1)]);
            }
        }
        let mut unique = Vec::with_capacity(groups.len());
        for group in groups {
            if !prefer_frequent_spelling {
                unique.push(
                    group
                        .into_iter()
                        .next()
                        .expect("each normalized source group has an entry")
                        .0,
                );
                continue;
            }
            let mut selected = group[0].0.clone();
            let mut highest_count = group[0].1;
            for (entry, count) in group.into_iter().skip(1) {
                if count > highest_count
                    || (count == highest_count
                        && entry.source.chars().cmp(selected.source.chars()).is_lt())
                {
                    selected = entry;
                    highest_count = count;
                }
            }
            unique.push(selected);
        }
        let matcher = if unique.is_empty() {
            None
        } else {
            if unique.iter().any(|entry| entry.source.is_empty()) {
                return Err(GlossaryError::EmptyPattern);
            }
            Some(
                AhoCorasickBuilder::new()
                    .match_kind(MatchKind::Standard)
                    .ascii_case_insensitive(true)
                    .build(unique.iter().map(|entry| entry.source.as_bytes()))
                    .map_err(|error| GlossaryError::Matcher(error.to_string()))?,
            )
        };
        Ok(Self {
            name: name.into(),
            entries: unique,
            matcher,
        })
    }

    pub fn active_entries_for_text(&self, text: &str) -> Vec<(String, String)> {
        let Some(matcher) = &self.matcher else {
            return Vec::new();
        };
        let normalized = normalize_match_text(text);
        let mut seen = std::collections::HashSet::new();
        matcher
            .find_overlapping_iter(normalized.as_bytes())
            .filter_map(|mat| {
                let index = mat.pattern().as_usize();
                seen.insert(index).then(|| {
                    (
                        self.entries[index].source.clone(),
                        self.entries[index].target.clone(),
                    )
                })
            })
            .collect()
    }

    pub fn to_csv(&self) -> String {
        let mut out = String::from("source,target,tgt_lng\r\n");
        for entry in &self.entries {
            out.push_str(&csv_field(&entry.source));
            out.push(',');
            out.push_str(&csv_field(&entry.target));
            out.push(',');
            out.push_str(&csv_field(entry.target_language.as_deref().unwrap_or("")));
            out.push_str("\r\n");
        }
        out
    }
}

pub fn normalize_source(source: &str) -> String {
    collapse_python_whitespace(source.to_lowercase().as_str())
}

fn normalize_match_text(text: &str) -> String {
    collapse_python_whitespace(text)
}

fn collapse_python_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if is_python_strip_whitespace(ch) {
            if !out.is_empty() {
                pending_space = true;
            }
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(ch);
        }
    }
    out
}

fn csv_field(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        let mut out = String::with_capacity(field.len() + 2);
        out.push('"');
        for ch in field.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
        out
    } else {
        field.to_owned()
    }
}

/// Preserves insertion-ordered source groups and chooses the codepoint-smallest tied target.
pub fn finalize_terms(
    name: impl Into<String>,
    raw_terms: &[(String, String)],
) -> Result<Option<Glossary>, GlossaryError> {
    if raw_terms.is_empty() {
        return Ok(None);
    }
    let mut source_indices: HashMap<&str, usize> = HashMap::new();
    let mut groups: Vec<(&str, usize, Vec<(&str, usize)>)> = Vec::new();
    for (source, target) in raw_terms {
        let index = if let Some(index) = source_indices.get(source.as_str()) {
            *index
        } else {
            let index = groups.len();
            source_indices.insert(source, index);
            groups.push((source, 0, Vec::new()));
            index
        };
        let group = &mut groups[index];
        group.1 += 1;
        let targets = &mut group.2;
        if let Some((_, count)) = targets
            .iter_mut()
            .find(|(candidate, _)| *candidate == target)
        {
            *count += 1;
        } else {
            targets.push((target, 1));
        }
    }
    let mut entries = Vec::with_capacity(raw_terms.len());
    for (source, source_count, targets) in groups {
        let mut targets = targets.into_iter();
        let (mut target, mut best_count) = targets.next().expect("each source group has a target");
        for (candidate, count) in targets {
            if count > best_count
                || (count == best_count && candidate.chars().cmp(target.chars()).is_lt())
            {
                target = candidate;
                best_count = count;
            }
        }
        for _ in 0..source_count {
            entries.push(GlossaryEntry::new(source, target));
        }
    }
    Glossary::new_with_source_spelling_votes(name, entries, true).map(Some)
}

#[cfg(test)]
mod glossary_tests;
