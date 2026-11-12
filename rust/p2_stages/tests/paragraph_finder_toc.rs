//! Exercise the production paragraph/style stages with synthetic glyph geometry.
use babeldoc_ir_core::{self as ir, Edge, Node, NodeData};
use babeldoc_p2_stages::shared::formula_helper::FontCoverage;
use babeldoc_p2_stages::stages::paragraph_finder;
use babeldoc_p2_stages::stages::paragraph_finder_inputs::{
    FontFace, FontFamily, FontMapper, ParagraphFinderConfig, ParagraphFinderInputs,
    PrimaryFontFamily, RandomDebugIdSource,
};
use babeldoc_p2_stages::stages::{
    styles_and_formulas, styles_and_formulas_config::StylesAndFormulasConfig,
};
use babeldoc_p2_stages::{NoopCancellation, NoopProgress, StageContext};
use std::sync::Arc;

fn edge(data: NodeData) -> Edge {
    Edge::Inline(Box::new(Node::new(data)))
}

fn bounds(x: f64, y: f64, x2: f64, y2: f64) -> Edge {
    edge(NodeData::Box(ir::DataBox {
        x: Some(x),
        y: Some(y),
        x2: Some(x2),
        y2: Some(y2),
    }))
}

fn document(rows: &[(String, bool)], dot_gap: f64) -> ir::Document {
    let mut characters = Vec::new();
    for (row, (text, page_number)) in rows.iter().enumerate() {
        let left = if *page_number { 500. } else { 50. };
        let y = 700. - row as f64 * 15.;
        let mut x = left;
        for ch in text.chars() {
            characters.push(edge(NodeData::PdfCharacter(ir::DataPdfCharacter {
                pdf_style: Some(edge(NodeData::PdfStyle(ir::DataPdfStyle {
                    font_id: Some("F1".into()),
                    font_size: Some(10.),
                    ..Default::default()
                }))),
                r#box: Some(bounds(x, y, x + 3., y + 10.)),
                visual_bbox: Some(edge(NodeData::VisualBbox(ir::DataVisualBbox {
                    r#box: Some(bounds(x, y, x + 3., y + 10.)),
                }))),
                char_unicode: Some(ch.to_string()),
                pdf_character_id: Some(ch as i64),
                xobj_id: Some(0),
                render_order: Some(characters.len() as i64),
                ..Default::default()
            })));
            x += 3. + if ch == '.' { dot_gap } else { 0. };
        }
    }
    let page = edge(NodeData::Page(ir::DataPage {
        page_number: Some(0),
        page_layout: vec![
            edge(NodeData::PageLayout(ir::DataPageLayout {
                id: Some(1),
                class_name: Some("plain text".into()),
                r#box: Some(bounds(40., 100., 450., 750.)),
                ..Default::default()
            })),
            edge(NodeData::PageLayout(ir::DataPageLayout {
                id: Some(2),
                class_name: Some("plain text".into()),
                r#box: Some(bounds(490., 100., 550., 750.)),
                ..Default::default()
            })),
        ],
        pdf_character: characters,
        ..Default::default()
    }));
    ir::Document::from_parts(
        Node::new(NodeData::Document(ir::DataDocument {
            page: vec![page],
            total_pages: Some(1),
        })),
        ir::Table::default(),
        ir::History::default(),
    )
    .unwrap()
}

fn font_mapper() -> FontMapper {
    // Paragraph finding does not read font metrics, but its input contract
    // requires a mapper. A minimal synthetic sfnt avoids system/font downloads.
    let mut sfnt = vec![0; 84];
    sfnt[..4].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    sfnt[4..6].copy_from_slice(&1u16.to_be_bytes());
    sfnt[12..16].copy_from_slice(b"head");
    sfnt[20..24].copy_from_slice(&28u32.to_be_bytes());
    sfnt[24..28].copy_from_slice(&54u32.to_be_bytes());
    sfnt[46..48].copy_from_slice(&1000u16.to_be_bytes());
    let path = std::env::temp_dir().join(format!(
        "babeldoc-toc-{}-{}.ttf",
        std::process::id(),
        fastrand::u64(..)
    ));
    std::fs::write(&path, sfnt).unwrap();
    let face = FontFace::load("test", &path);
    std::fs::remove_file(path).unwrap();
    FontMapper::new(
        FontFamily {
            base: vec![face.unwrap()],
            ..Default::default()
        },
        PrimaryFontFamily::None,
    )
    .unwrap()
}

fn find_paragraphs(rows: &[(String, bool)]) -> ir::Document {
    find_document_paragraphs(document(rows, 0.))
}

fn find_document_paragraphs(mut document: ir::Document) -> ir::Document {
    let config = ParagraphFinderConfig::default();
    let mapper = font_mapper();
    let mut ids = RandomDebugIdSource;
    let mut inputs = ParagraphFinderInputs {
        config: &config,
        font_mapper: &mapper,
        debug_ids: &mut ids,
    };
    let mut context = StageContext::new(&NoopCancellation, &NoopProgress);
    paragraph_finder::process_with(&mut inputs, &mut document, &mut context).unwrap();
    document
}

fn texts(document: &ir::Document) -> Vec<String> {
    let view = document.read();
    let page = view.document().page_at(&view, 0).unwrap();
    page.pdf_paragraph_iter(&view)
        .unwrap()
        .map(|paragraph| paragraph.unwrap().unicode(&view).unwrap().clone().unwrap())
        .collect()
}

#[test]
fn directory_entries_split_for_contiguous_and_spaced_dot_leaders() {
    for separator in ["", " ", "\t", "\u{a0}", "\u{2009}", "\u{202f}"] {
        let minimum = if separator.is_empty() { 20 } else { 10 };
        let leader = vec!["."; minimum].join(separator);
        let rows: Vec<_> = (1..=5)
            .map(|index| {
                (
                    format!(
                        "3.{index} Coordinate transformation x {leader} {}",
                        index + 5
                    ),
                    false,
                )
            })
            .collect();
        let document = find_paragraphs(&rows);
        let paragraphs = texts(&document);
        assert_eq!(paragraphs.len(), 5, "separator {separator:?}");
        for (index, text) in paragraphs.iter().enumerate() {
            assert!(text.starts_with(&format!("3.{} ", index + 1)), "{text:?}");
        }
        let view = document.read();
        let page = view.document().page_at(&view, 0).unwrap();
        for (index, paragraph) in page.pdf_paragraph_iter(&view).unwrap().enumerate() {
            let paragraph = paragraph.unwrap();
            assert_eq!(paragraph.pdf_paragraph_composition_len(&view).unwrap(), 1);
            let bounds = paragraph.r#box(&view).unwrap().unwrap();
            assert_eq!(bounds.y(&view).unwrap(), Some(700. - index as f64 * 15.));
            assert_eq!(bounds.y2(&view).unwrap(), Some(710. - index as f64 * 15.));
            assert_eq!(paragraph.xobj_id(&view).unwrap(), Some(0));
            assert!(paragraph.render_order(&view).unwrap().is_some());
        }
    }
}

#[test]
fn geometry_inserted_spaces_in_dot_leaders_still_split() {
    let rows = [
        (format!("Introduction {} 2", ".".repeat(10)), false),
        (format!("Definitions {} 4", ".".repeat(10)), false),
    ];
    let document = find_document_paragraphs(document(&rows, 2.));
    let paragraphs = texts(&document);
    assert_eq!(paragraphs.len(), 2);
    assert!(paragraphs.iter().all(|text| text.contains(". .")));
}

#[test]
fn dot_leader_threshold_depends_on_internal_horizontal_spacing() {
    for separator in ["", " ", "\t", "\u{a0}", "\u{2009}", "\u{202f}"] {
        for count in [9, 10, 19, 20] {
            let leader = vec!["."; count].join(separator);
            let expected = if count >= if separator.is_empty() { 20 } else { 10 } {
                2
            } else {
                1
            };
            let document = find_paragraphs(&[
                (format!("Entry {leader} 6"), false),
                ("Following line".into(), false),
            ]);
            assert_eq!(
                texts(&document).len(),
                expected,
                "{count} dots, separator {separator:?}"
            );
        }
    }
    for (leader, expected) in [
        (format!("{} .", ".".repeat(9)), 2),
        (format!(". {}", ".".repeat(9)), 2),
        (format!("{} .", ".".repeat(8)), 1),
        (format!("{} ", ".".repeat(19)), 1),
        (format!(" {}", ".".repeat(19)), 1),
        (format!("{}\n{}", ".".repeat(5), ".".repeat(5)), 1),
        (format!("{}\u{2028}{}", ".".repeat(5), ".".repeat(5)), 1),
    ] {
        let document = find_paragraphs(&[
            (format!("Entry {leader} 6"), false),
            ("Following line".into(), false),
        ]);
        assert_eq!(texts(&document).len(), expected, "{leader:?}");
    }
}

#[test]
fn ordinary_wrapping_and_short_ellipses_do_not_split() {
    for first in [
        "A sentence with an ellipsis ... and more words".to_owned(),
        format!("Only nine spaced dots {} 6", ". ".repeat(9)),
        format!("Only nineteen contiguous dots {} 6", ".".repeat(19)),
        format!("Separate runs {} word {} 6", ".".repeat(10), ".".repeat(10)),
        format!(
            "Vertical whitespace {}\n{} 6",
            ".".repeat(10),
            ".".repeat(10)
        ),
    ] {
        let document = find_paragraphs(&[
            (first.clone(), false),
            ("Continued body text".into(), false),
        ]);
        assert_eq!(texts(&document).len(), 1, "{first:?}");
    }
}

#[test]
fn page_number_paragraphs_do_not_rejoin_directory_entries() {
    for separator in ["", " ", "\u{a0}"] {
        let minimum = if separator.is_empty() { 20 } else { 10 };
        let leader = vec!["."; minimum].join(separator);
        for (first, second) in [
            (
                format!("1 Introduction {leader}"),
                format!("2 Definitions {leader}"),
            ),
            ("1 Introduction".into(), format!("2 Definitions {leader}")),
            (format!("1 Introduction {leader}"), "2 Definitions".into()),
        ] {
            let rows = [
                (first, false),
                ("2".into(), true),
                (second, false),
                ("4".into(), true),
            ];
            let document = find_paragraphs(&rows);
            let paragraphs = texts(&document);
            assert_eq!(paragraphs.len(), 4, "separator {separator:?}");
            assert!(paragraphs[0].starts_with("1 Introduction"));
            assert_eq!(paragraphs[1], "2");
            assert!(paragraphs[2].starts_with("2 Definitions"));
            assert_eq!(paragraphs[3], "4");
        }
    }
}

#[test]
fn alternating_body_line_numbers_still_merge() {
    let document = find_paragraphs(&[
        ("The first body line".into(), false),
        ("12".into(), true),
        ("The second body line".into(), false),
    ]);
    let paragraphs = texts(&document);
    assert_eq!(paragraphs.len(), 2);
    assert!(paragraphs[0].contains("first body line"));
    assert!(paragraphs[0].contains("second body line"));
    assert_eq!(paragraphs[1], "12");
}

#[test]
fn dot_leader_fragments_on_distinct_lines_do_not_block_body_merging() {
    for (separator, count) in [("", 10), (" ", 5), ("\u{a0}", 5)] {
        let fragment = vec!["."; count].join(separator);
        for fragments_first in [true, false] {
            // Each physical line is below its leader threshold; only flattening
            // the two line compositions would create a recognized dot leader.
            let wrapped = [
                (format!("Wrapped body {fragment}"), false),
                (format!("{fragment} continues"), false),
            ];
            let rows = if fragments_first {
                [
                    wrapped[0].clone(),
                    wrapped[1].clone(),
                    ("12".into(), true),
                    ("Continued body".into(), false),
                ]
            } else {
                [
                    ("Previous body".into(), false),
                    ("12".into(), true),
                    wrapped[0].clone(),
                    wrapped[1].clone(),
                ]
            };
            let document = find_paragraphs(&rows);
            let paragraphs = texts(&document);
            assert_eq!(
                paragraphs.len(),
                2,
                "separator {separator:?}, fragments_first {fragments_first}"
            );
            assert!(paragraphs[0].contains("Wrapped body"));
            assert!(paragraphs[0].contains("continues"));
            assert_eq!(paragraphs[1], "12");

            let view = document.read();
            let page = view.document().page_at(&view, 0).unwrap();
            let body = page.pdf_paragraph_at(&view, 0).unwrap();
            assert_eq!(body.pdf_paragraph_composition_len(&view).unwrap(), 3);
            let expected_y = if fragments_first {
                [700., 685., 655.]
            } else {
                [700., 670., 655.]
            };
            for (index, y) in expected_y.into_iter().enumerate() {
                let composition = body.pdf_paragraph_composition_at(&view, index).unwrap();
                let line = composition.pdf_line(&view).unwrap().unwrap();
                let bounds = line.r#box(&view).unwrap().unwrap();
                assert_eq!(bounds.y(&view).unwrap(), Some(y));
                assert_eq!(bounds.y2(&view).unwrap(), Some(y + 10.));
            }
        }
    }
}

#[test]
fn directory_rows_and_formulas_survive_style_grouping() {
    struct Covered;
    impl FontCoverage for Covered {
        fn has_char(&self, _: &str) -> bool {
            true
        }
    }
    let rows: Vec<_> = (1..=5)
        .map(|index| {
            (
                format!(
                    "3.{index} Coordinate transformation x {} {}",
                    ". ".repeat(10),
                    index + 5
                ),
                false,
            )
        })
        .collect();
    let mut document = find_paragraphs(&rows);
    let before = texts(&document);
    assert_eq!(before.len(), 5);
    let config = StylesAndFormulasConfig {
        font_mapper: Arc::new(Covered),
        formula_matcher: None,
        character_pattern_hits: Some([("x".into(), true)].into()),
        font_pattern_hits: None,
        skip_formula_offset_calculation: false,
        ocr_workaround: false,
        remove_non_formula_lines: false,
        figure_table_protection_threshold: None,
        non_formula_line_iou_threshold: None,
    };
    let mut context = StageContext::new(&NoopCancellation, &NoopProgress);
    styles_and_formulas::process(&config, &mut document, &mut context).unwrap();
    assert_eq!(texts(&document), before);
    let view = document.read();
    let page = view.document().page_at(&view, 0).unwrap();
    for (index, paragraph) in page.pdf_paragraph_iter(&view).unwrap().enumerate() {
        let paragraph = paragraph.unwrap();
        let mut formulas = Vec::new();
        for composition in paragraph.pdf_paragraph_composition_iter(&view).unwrap() {
            if let Some(formula) = composition.unwrap().pdf_formula(&view).unwrap() {
                formulas.push(formula);
            }
        }
        assert_eq!(formulas.len(), 1);
        let formula = &formulas[0];
        let formula_text: String = formula
            .pdf_character_iter(&view)
            .unwrap()
            .map(|character| {
                character
                    .unwrap()
                    .char_unicode(&view)
                    .unwrap()
                    .clone()
                    .unwrap()
            })
            .collect();
        assert_eq!(formula_text.trim(), "x");
        let character = formula.pdf_character_at(&view, 0).unwrap();
        assert_eq!(character.char_unicode(&view).unwrap().as_deref(), Some("x"));
        let bounds = character.r#box(&view).unwrap().unwrap();
        assert_eq!(bounds.y(&view).unwrap(), Some(700. - index as f64 * 15.));
        let style = character.pdf_style(&view).unwrap().unwrap();
        assert_eq!(style.font_size(&view).unwrap(), Some(10.));
    }
}
