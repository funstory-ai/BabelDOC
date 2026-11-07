use babeldoc_il_types::schema::{PdfParagraph, PdfParagraphComposition};

use crate::{
    il_types::schema::PdfCharacter, shared::layout_helper::unicode::is_python_strip_whitespace,
};

pub fn is_cid_paragraph(paragraph: &PdfParagraph) -> bool {
    let mut chars: Vec<&PdfCharacter> = Vec::new();
    for composition in &paragraph.pdf_paragraph_composition {
        if let Some(line) = &composition.pdf_line {
            chars.extend(line.pdf_character.iter());
        } else if let Some(same_style) = &composition.pdf_same_style_characters {
            chars.extend(same_style.pdf_character.iter());
        } else if composition.pdf_same_style_unicode_characters.is_some() {
            continue;
        } else if let Some(formula) = &composition.pdf_formula {
            chars.extend(formula.pdf_character.iter());
        } else if let Some(character) = &composition.pdf_character {
            chars.push(character);
        }
    }
    let cid_count = chars
        .iter()
        .filter(|character| character.char_unicode.as_deref().is_some_and(is_cid_text))
        .count();
    cid_count * 5 > chars.len() * 4
}

fn is_cid_text(text: &str) -> bool {
    text.strip_prefix("(cid:")
        .and_then(|tail| tail.strip_suffix(')'))
        .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

pub fn is_pure_numeric_paragraph(paragraph: &PdfParagraph) -> bool {
    let Some(text) = paragraph.unicode.as_deref() else {
        return false;
    };
    is_pure_numeric_text(text)
}

pub fn is_pure_numeric_text(text: &str) -> bool {
    let text = trim_python_whitespace(text);
    if text.is_empty() {
        return false;
    }
    let text = text.strip_prefix('-').unwrap_or(text);
    if text.is_empty() {
        return false;
    }
    let mut parts = text.split('.');
    let integer = parts.next().unwrap_or_default();
    let fraction = parts.next();
    parts.next().is_none()
        && !integer.is_empty()
        && integer.chars().all(is_decimal_digit)
        && fraction
            .is_none_or(|fraction| !fraction.is_empty() && fraction.chars().all(is_decimal_digit))
}

pub fn is_placeholder_only_paragraph(paragraph: &PdfParagraph) -> bool {
    if paragraph.unicode.as_deref().is_none_or(str::is_empty) {
        return false;
    }
    paragraph
        .pdf_paragraph_composition
        .iter()
        .all(composition_is_placeholder_or_space)
}

fn composition_is_placeholder_or_space(composition: &PdfParagraphComposition) -> bool {
    if composition.pdf_formula.is_some() {
        true
    } else if let Some(character) = &composition.pdf_character {
        character
            .char_unicode
            .as_deref()
            .is_some_and(python_isspace)
    } else if let Some(line) = &composition.pdf_line {
        line.pdf_character.iter().all(|character| {
            character
                .char_unicode
                .as_deref()
                .is_some_and(python_isspace)
        })
    } else if let Some(group) = &composition.pdf_same_style_characters {
        group.pdf_character.iter().all(|character| {
            character
                .char_unicode
                .as_deref()
                .is_some_and(python_isspace)
        })
    } else if let Some(group) = &composition.pdf_same_style_unicode_characters {
        group.unicode.as_deref().is_some_and(python_isspace)
    } else {
        false
    }
}

fn python_isspace(text: &str) -> bool {
    !text.is_empty() && text.chars().all(is_python_strip_whitespace)
}

fn trim_python_whitespace(text: &str) -> &str {
    let start = text
        .char_indices()
        .find(|(_, ch)| !is_python_strip_whitespace(*ch))
        .map_or(text.len(), |(i, _)| i);
    let end = text
        .char_indices()
        .rev()
        .find(|(_, ch)| !is_python_strip_whitespace(*ch))
        .map_or(start, |(i, ch)| i + ch.len_utf8());
    &text[start..end]
}

// Unicode General_Category=Nd ranges from the Python 3.13 runtime used by this port.
fn is_decimal_digit(ch: char) -> bool {
    const RANGES: &[(u32, u32)] = &[
        (0x30, 0x39),
        (0x660, 0x669),
        (0x6f0, 0x6f9),
        (0x7c0, 0x7c9),
        (0x966, 0x96f),
        (0x9e6, 0x9ef),
        (0xa66, 0xa6f),
        (0xae6, 0xaef),
        (0xb66, 0xb6f),
        (0xbe6, 0xbef),
        (0xc66, 0xc6f),
        (0xce6, 0xcef),
        (0xd66, 0xd6f),
        (0xde6, 0xdef),
        (0xe50, 0xe59),
        (0xed0, 0xed9),
        (0xf20, 0xf29),
        (0x1040, 0x1049),
        (0x1090, 0x1099),
        (0x17e0, 0x17e9),
        (0x1810, 0x1819),
        (0x1946, 0x194f),
        (0x19d0, 0x19d9),
        (0x1a80, 0x1a89),
        (0x1a90, 0x1a99),
        (0x1b50, 0x1b59),
        (0x1bb0, 0x1bb9),
        (0x1c40, 0x1c49),
        (0x1c50, 0x1c59),
        (0xa620, 0xa629),
        (0xa8d0, 0xa8d9),
        (0xa900, 0xa909),
        (0xa9d0, 0xa9d9),
        (0xa9f0, 0xa9f9),
        (0xaa50, 0xaa59),
        (0xabf0, 0xabf9),
        (0xff10, 0xff19),
        (0x104a0, 0x104a9),
        (0x10d30, 0x10d39),
        (0x11066, 0x1106f),
        (0x110f0, 0x110f9),
        (0x11136, 0x1113f),
        (0x111d0, 0x111d9),
        (0x112f0, 0x112f9),
        (0x11450, 0x11459),
        (0x114d0, 0x114d9),
        (0x11650, 0x11659),
        (0x116c0, 0x116c9),
        (0x11730, 0x11739),
        (0x118e0, 0x118e9),
        (0x11950, 0x11959),
        (0x11c50, 0x11c59),
        (0x11d50, 0x11d59),
        (0x11da0, 0x11da9),
        (0x11f50, 0x11f59),
        (0x16a60, 0x16a69),
        (0x16ac0, 0x16ac9),
        (0x16b50, 0x16b59),
        (0x1d7ce, 0x1d7ff),
        (0x1e140, 0x1e149),
        (0x1e2f0, 0x1e2f9),
        (0x1e4f0, 0x1e4f9),
        (0x1e950, 0x1e959),
        (0x1fbf0, 0x1fbf9),
    ];
    let code = ch as u32;
    RANGES
        .iter()
        .any(|(start, end)| (*start..=*end).contains(&code))
}

#[cfg(test)]
mod paragraph_helpers_tests;
