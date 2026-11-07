//! Scalar inputs for pure geometry/unicode helpers; never a second object graph.
use crate::StageError;
use babeldoc_il_types::{Box as B, PageLayout, PdfCharacter, PdfStyle, VisualBbox};
use babeldoc_ir_core::{self as ir, Cursor, Permission, TreeView, types};
pub fn need<T>(v: Option<T>, s: &str) -> Result<T, StageError> {
    v.ok_or_else(|| StageError::failed(s))
}
pub fn box_read<V: Permission>(
    c: &Cursor<types::Box>,
    v: &TreeView<'_, V>,
) -> Result<B, StageError> {
    Ok(B {
        x: c.x(v)?,
        y: c.y(v)?,
        x2: c.x2(v)?,
        y2: c.y2(v)?,
    })
}
pub fn box_value(b: B) -> ir::FieldValue<types::Box> {
    ir::BoxInit {
        x: b.x,
        y: b.y,
        x2: b.x2,
        y2: b.y2,
    }
    .into_value()
}
pub fn visual<V: Permission>(
    c: &Cursor<types::PdfCharacter>,
    v: &TreeView<'_, V>,
) -> Result<B, StageError> {
    box_read(
        &need(
            need(c.visual_bbox(v)?, "visual_bbox")?.r#box(v)?,
            "visual_bbox.box",
        )?,
        v,
    )
}
pub fn character<V: Permission>(
    c: &Cursor<types::PdfCharacter>,
    v: &TreeView<'_, V>,
) -> Result<PdfCharacter, StageError> {
    Ok(PdfCharacter {
        r#box: c
            .r#box(v)?
            .map(|b| box_read(&b, v).map(Box::new))
            .transpose()?,
        visual_bbox: Some(Box::new(VisualBbox {
            r#box: Some(Box::new(visual(c, v)?)),
        })),
        pdf_style: c
            .pdf_style(v)?
            .map(|s| {
                Ok::<_, StageError>(Box::new(PdfStyle {
                    font_id: s.font_id(v)?.clone(),
                    font_size: s.font_size(v)?,
                    graphic_state: None,
                }))
            })
            .transpose()?,
        pdf_character_id: c.pdf_character_id(v)?,
        char_unicode: c.char_unicode(v)?.clone(),
        scale: c.scale(v)?,
        vertical: c.vertical(v)?,
        advance: c.advance(v)?,
        xobj_id: c.xobj_id(v)?,
        formula_layout_id: c.formula_layout_id(v)?,
        render_order: c.render_order(v)?,
        ..Default::default()
    })
}
pub fn layouts<V: Permission>(
    page: &Cursor<types::Page>,
    v: &TreeView<'_, V>,
) -> Result<Vec<PageLayout>, StageError> {
    page.page_layout_iter(v)?
        .map(|c| {
            let c = c?;
            Ok(PageLayout {
                id: c.id(v)?,
                class_name: c.class_name(v)?.clone(),
                r#box: c
                    .r#box(v)?
                    .map(|b| box_read(&b, v).map(Box::new))
                    .transpose()?,
                ..Default::default()
            })
        })
        .collect()
}
pub fn union(boxes: &[B]) -> Result<B, StageError> {
    let mut result = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for b in boxes {
        result.0 = result.0.min(need(b.x, "box.x")?);
        result.1 = result.1.min(need(b.y, "box.y")?);
        result.2 = result.2.max(need(b.x2, "box.x2")?);
        result.3 = result.3.max(need(b.y2, "box.y2")?);
    }
    Ok(B {
        x: Some(result.0),
        y: Some(result.1),
        x2: Some(result.2),
        y2: Some(result.3),
    })
}
