//! Debug annotations preserve Python box, style and palette identities.
use crate::StageError;
use crate::shared::style::{BLUE, ORANGE, PINK, PaletteColor, TEAL, YELLOW};
use babeldoc_ir_core::{self as ir, AddDebugInformationView, Cursor, FieldValue, TreeView, types};
use std::collections::HashMap;
type View<'a> = TreeView<'a, AddDebugInformationView>;
type Page = Cursor<types::Page>;
type B = Cursor<types::Box>;
fn need<T>(value: Option<T>, name: &str) -> Result<T, StageError> {
    value.ok_or_else(|| StageError::failed(name))
}
struct Palette {
    selections: HashMap<usize, Cursor<types::GraphicState>>,
}
impl Palette {
    fn value(
        &self,
        color: &'static PaletteColor,
        v: &mut View<'_>,
    ) -> Result<FieldValue<types::GraphicState>, StageError> {
        if let Some(c) = self.selections.get(&(color as *const _ as usize)) {
            return Ok(c.share(v)?);
        }
        let value = color.get();
        let instruction = match value.as_ref().passthrough_per_char_instruction.as_ref() {
            Some(babeldoc_il_types::PassthroughValue::Materialized(s)) => s.clone(),
            _ => return Err(StageError::failed("palette must be materialized")),
        };
        Ok(ir::GraphicStateInit {
            passthrough_per_char_instruction: Some(ir::Instructions::Materialized(instruction)),
            ..Default::default()
        }
        .into_value())
    }
    fn remember(
        &mut self,
        color: &'static PaletteColor,
        c: Cursor<types::GraphicState>,
        v: &mut View<'_>,
    ) -> Result<(), StageError> {
        self.selections
            .entry(color as *const _ as usize)
            .or_insert(c.retain(v)?);
        Ok(())
    }
}
fn text(
    page: &Page,
    v: &mut View<'_>,
    palette: &mut Palette,
    value: String,
    color: &'static PaletteColor,
    coords: (Option<f64>, Option<f64>, Option<f64>),
    font_size: f64,
) -> Result<(), StageError> {
    let (x, x2, y2) = coords;
    let y2 = need(y2, "debug text box.y2")?;
    let state = palette.value(color, v)?;
    let index = page.pdf_paragraph_len(v)?;
    page.push_pdf_paragraph(
        v,
        ir::PdfParagraphInit {
            first_line_indent: Some(false),
            vertical: Some(false),
            xobj_id: Some(-1),
            unicode: Some(value.clone()),
            r#box: Some(
                ir::BoxInit {
                    x,
                    y: Some(y2),
                    x2,
                    y2: Some(y2 + 5.),
                }
                .into_value(),
            ),
            pdf_style: Some(
                ir::PdfStyleInit {
                    font_id: Some("base".into()),
                    font_size: Some(font_size),
                    graphic_state: Some(state),
                }
                .into_value(),
            ),
            ..Default::default()
        }
        .into_value(),
    )?;
    let paragraph = page.pdf_paragraph_at(v, index)?;
    let style = need(paragraph.pdf_style(v)?, "debug paragraph style")?;
    palette.remember(
        color,
        need(style.graphic_state(v)?, "debug graphic state")?,
        v,
    )?;
    let shared_style = style.share(v)?;
    paragraph.push_pdf_paragraph_composition(
        v,
        ir::PdfParagraphCompositionInit {
            pdf_same_style_unicode_characters: Some(
                ir::PdfSameStyleUnicodeCharactersInit {
                    unicode: Some(value),
                    pdf_style: Some(shared_style),
                    debug_info: Some(true),
                }
                .into_value(),
            ),
            ..Default::default()
        }
        .into_value(),
    )?;
    Ok(())
}
fn coords(b: &B, v: &View<'_>) -> Result<(Option<f64>, Option<f64>, Option<f64>), StageError> {
    Ok((b.x(v)?, b.x2(v)?, b.y2(v)?))
}
fn rectangle(
    page: &Page,
    v: &mut View<'_>,
    palette: &mut Palette,
    b: B,
    color: &'static PaletteColor,
    width: Option<f64>,
) -> Result<(), StageError> {
    let state = palette.value(color, v)?;
    let b = b.share(v)?;
    let i = page.pdf_rectangle_len(v)?;
    page.push_pdf_rectangle(
        v,
        ir::PdfRectangleInit {
            r#box: Some(b),
            graphic_state: Some(state),
            debug_info: Some(true),
            line_width: width,
            ..Default::default()
        }
        .into_value(),
    )?;
    let c = need(
        page.pdf_rectangle_at(v, i)?.graphic_state(v)?,
        "debug graphic state",
    )?;
    palette.remember(color, c, v)
}
pub fn process(debug: bool, document: &mut ir::Document) -> Result<(), StageError> {
    if !debug {
        return Ok(());
    }
    let mut view = document.view::<AddDebugInformationView>();
    let mut palette = Palette {
        selections: HashMap::new(),
    };
    for page_index in 0..view.document().page_len(&view)? {
        let page = view.document().page_at(&view, page_index)?;
        let b = need(
            need(page.cropbox(&view)?, "page.cropbox")?.r#box(&view)?,
            "page.cropbox.box",
        )?;
        let x = need(b.x(&view)?, "cropbox.x")?;
        let y = need(b.y(&view)?, "cropbox.y")?;
        let x2 = need(b.x2(&view)?, "cropbox.x2")?;
        let y2 = need(b.y2(&view)?, "cropbox.y2")?;
        let number = need(page.page_number(&view)?, "page_number")? + 1;
        text(
            &page,
            &mut view,
            &mut palette,
            format!("pagenumber: {number}"),
            &BLUE,
            (
                Some(x + (x2 - x) * 0.02),
                Some(x2),
                Some(y2 - (y2 - y) * 0.02),
            ),
            4.,
        )?;
        let xobject_len = page.pdf_xobject_len(&view)?;
        let form_len = page.pdf_form_len(&view)?;
        // Adding debug paragraphs mutates this collection, so preserve its start-of-pass snapshot.
        let paragraphs = page
            .pdf_paragraph_iter(&view)?
            .collect::<Result<Vec<_>, _>>()?;
        for paragraph in paragraphs {
            let composition_len = paragraph.pdf_paragraph_composition_len(&view)?;
            if composition_len == 0 {
                continue;
            }
            let mut debug = false;
            for child_index in 0..composition_len {
                let c = paragraph.pdf_paragraph_composition_at(&view, child_index)?;
                if let Some(s) = c.pdf_same_style_unicode_characters(&view)? {
                    debug |= s.debug_info(&view)?.unwrap_or(false);
                }
            }
            if debug {
                continue;
            }
            let b = need(paragraph.r#box(&view)?, "paragraph.box")?;
            let bc = coords(&b, &view)?;
            rectangle(&page, &mut view, &mut palette, b, &BLUE, None)?;
            let label = match paragraph
                .debug_id(&view)?
                .as_deref()
                .filter(|s| !s.is_empty())
            {
                Some(id) => format!(
                    "paragraph[{id}]-[{}]",
                    paragraph.layout_label(&view)?.as_deref().unwrap_or("None")
                ),
                None => "paragraph".into(),
            };
            text(&page, &mut view, &mut palette, label, &BLUE, bc, 4.)?;
            for child_index in 0..composition_len {
                let c = paragraph.pdf_paragraph_composition_at(&view, child_index)?;
                if let Some(formula) = c.pdf_formula(&view)? {
                    let b = need(formula.r#box(&view)?, "formula.box")?;
                    let bc = coords(&b, &view)?;
                    text(
                        &page,
                        &mut view,
                        &mut palette,
                        "formula".into(),
                        &ORANGE,
                        bc,
                        4.,
                    )?;
                    rectangle(&page, &mut view, &mut palette, b, &ORANGE, None)?;
                    for child_index in 0..formula.pdf_character_len(&view)? {
                        let character = formula.pdf_character_at(&view, child_index)?;
                        let b = need(
                            need(character.visual_bbox(&view)?, "visual_bbox")?.r#box(&view)?,
                            "visual_bbox.box",
                        )?;
                        rectangle(&page, &mut view, &mut palette, b, &TEAL, Some(0.2))?;
                    }
                }
            }
            for child_index in 0..xobject_len {
                let xo = page.pdf_xobject_at(&view, child_index)?;
                let b = need(xo.r#box(&view)?, "xobject.box")?;
                rectangle(&page, &mut view, &mut palette, b, &YELLOW, None)?;
            }
            for child_index in 0..form_len {
                let form = page.pdf_form_at(&view, child_index)?;
                let subtype = need(form.pdf_form_subtype(&view)?, "form subtype")?;
                let label = if let Some(xo) = subtype.pdf_xobj_form(&view)? {
                    format!("Form[{}]", xo.do_args(&view)?.as_deref().unwrap_or("None"))
                } else if subtype.pdf_inline_form(&view)?.is_some() {
                    "Form[inline]".into()
                } else {
                    "Form".into()
                };
                let b = need(form.r#box(&view)?, "form.box")?;
                let bc = coords(&b, &view)?;
                text(&page, &mut view, &mut palette, label, &PINK, bc, 0.4)?;
                rectangle(&page, &mut view, &mut palette, b, &PINK, None)?;
            }
        }
    }
    drop(palette);
    drop(view);
    document.compact();
    Ok(())
}
