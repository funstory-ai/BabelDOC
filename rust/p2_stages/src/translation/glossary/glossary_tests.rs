use super::*;

fn glossary(entries: &[(&str, &str)]) -> Glossary {
    let entries = entries
        .iter()
        .map(|(source, target)| GlossaryEntry::new(*source, *target))
        .collect();
    Glossary::new("test_glossary", entries).expect("glossary should be valid")
}

fn sources(entries: &[(String, String)]) -> Vec<String> {
    entries.iter().map(|(source, _)| source.clone()).collect()
}

#[test]
fn empty_glossary_matches_nothing() {
    let glossary = glossary(&[]);
    assert!(glossary.active_entries_for_text("any text").is_empty());
}

#[test]
fn standalone_term_matches() {
    let glossary = glossary(&[("herkogamy", "雌雄异位")]);
    let hits = glossary.active_entries_for_text("herkogamy and dichogamy are separate");
    assert_eq!(sources(&hits), vec!["herkogamy".to_string()]);
}

#[test]
fn standalone_term_matches_case_insensitively() {
    let glossary = glossary(&[("herkogamy", "雌雄异位")]);
    let hits = glossary.active_entries_for_text("Herkogamy is a mechanism");
    assert_eq!(sources(&hits), vec!["herkogamy".to_string()]);
}

#[test]
fn term_is_not_matched_inside_a_longer_word() {
    let glossary = glossary(&[("etal", "甲")]);
    assert!(
        glossary
            .active_entries_for_text("the metal surface")
            .is_empty()
    );
    assert!(
        glossary
            .active_entries_for_text("a petal of the flower")
            .is_empty()
    );
    assert!(
        glossary
            .active_entries_for_text("metallic petals")
            .is_empty()
    );
}

#[test]
fn term_is_not_matched_inside_a_longer_word_case_insensitively() {
    let glossary = glossary(&[("etal", "甲")]);
    assert!(
        glossary
            .active_entries_for_text("the Metal surface")
            .is_empty()
    );
}

#[test]
fn term_is_matched_at_the_start_and_at_the_end_of_the_text() {
    let glossary = glossary(&[("etal", "甲")]);
    assert_eq!(
        sources(&glossary.active_entries_for_text("etal is a prefix")),
        vec!["etal".to_string()]
    );
    assert_eq!(
        sources(&glossary.active_entries_for_text("this ends with etal")),
        vec!["etal".to_string()]
    );
}

#[test]
fn term_is_matched_next_to_punctuation() {
    let glossary = glossary(&[("etal", "甲")]);
    assert_eq!(
        sources(&glossary.active_entries_for_text("etal, and (etal) too")),
        vec!["etal".to_string()]
    );
}

#[test]
fn term_is_not_matched_across_a_word_boundary() {
    let glossary = glossary(&[("etal", "甲")]);
    assert!(
        glossary
            .active_entries_for_text("etal2 is a different token")
            .is_empty()
    );
    assert!(
        glossary
            .active_entries_for_text("xetal is not a word")
            .is_empty()
    );
}

#[test]
fn longer_term_still_matches_when_a_shorter_term_is_embedded() {
    let glossary = glossary(&[("etal", "甲"), ("metal", "金属")]);
    let hits = glossary.active_entries_for_text("the metal surface");
    assert_eq!(sources(&hits), vec!["metal".to_string()]);
}

#[test]
fn multi_word_term_matches() {
    let glossary = glossary(&[("cross pollination", "异花传粉")]);
    assert_eq!(
        sources(&glossary.active_entries_for_text("the cross pollination process")),
        vec!["cross pollination".to_string()]
    );
    assert!(
        glossary
            .active_entries_for_text("crosspollination process")
            .is_empty()
    );
}

#[test]
fn cjk_term_is_not_restricted_by_ascii_word_boundaries() {
    let glossary = glossary(&[("金属", "metal")]);
    assert_eq!(
        sources(&glossary.active_entries_for_text("黄金金属的表面")),
        vec!["金属".to_string()]
    );
}

#[test]
fn term_with_non_alphanumeric_edge_keeps_its_own_boundaries() {
    let glossary = glossary(&[("C++", "C++")]);
    assert_eq!(
        sources(&glossary.active_entries_for_text("written in C++ today")),
        vec!["C++".to_string()]
    );
    assert!(glossary.active_entries_for_text("CC++ today").is_empty());
}

#[test]
fn each_entry_is_reported_once() {
    let glossary = glossary(&[("etal", "甲")]);
    let hits = glossary.active_entries_for_text("etal and etal again");
    assert_eq!(hits.len(), 1);
}

#[test]
fn normalize_source_lowercases_and_collapses_whitespace() {
    assert_eq!(normalize_source("  Herkogamy\t"), "herkogamy");
    assert_eq!(normalize_source("Cross   Pollination"), "cross pollination");
}

#[test]
fn duplicate_sources_are_collapsed() {
    let glossary = glossary(&[("Herkogamy", "a"), ("herkogamy", "b")]);
    assert_eq!(glossary.entries.len(), 1);
}

#[test]
fn to_csv_escapes_separators() {
    let glossary = glossary(&[("etal", "a,b")]);
    let csv = glossary.to_csv();
    assert!(csv.starts_with("source,target,tgt_lng\r\n"));
    assert!(csv.contains("\"a,b\""));
}

#[test]
fn finalize_terms_prefers_the_most_frequent_target() {
    let glossary = finalize_terms(
        "test",
        &[
            ("etal".to_string(), "甲".to_string()),
            ("etal".to_string(), "乙".to_string()),
            ("etal".to_string(), "甲".to_string()),
        ],
    )
    .expect("terms should be finalized")
    .expect("terms should not be empty");
    let hits = glossary.active_entries_for_text("etal");
    assert_eq!(hits, vec![("etal".to_string(), "甲".to_string())]);
}
