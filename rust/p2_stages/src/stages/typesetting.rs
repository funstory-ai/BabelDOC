//! Serial typesetting on M IR. Geometry trials use private numeric caches.
//! Document navigation and writes go exclusively through generated access.
use crate::{StageContext, StageError};
use babeldoc_font_core::{FontFace, FontMapper, SourceFontStyle};
use babeldoc_il_types::schema::Box;
use babeldoc_ir_core::{self as ir, Cursor, FieldValue, TreeView, TypesettingView, types};
use log::{error, warn};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
const LOG_TARGET: &str = "babeldoc.format.pdf.document_il.midend.typesetting";
type View<'a> = TreeView<'a, TypesettingView>;
type P = Cursor<types::PdfParagraph>;
type Page = Cursor<types::Page>;
fn need<T>(v: Option<T>, s: &str) -> Result<T, StageError> {
    v.ok_or_else(|| StageError::failed(s))
}
fn read_box(c: Cursor<types::Box>, v: &View<'_>) -> Result<Box, StageError> {
    Ok(Box {
        x: c.x(v)?,
        y: c.y(v)?,
        x2: c.x2(v)?,
        y2: c.y2(v)?,
    })
}
fn box_field(b: Box) -> FieldValue<types::Box> {
    ir::BoxInit {
        x: b.x,
        y: b.y,
        x2: b.x2,
        y2: b.y2,
    }
    .into_value()
}
// These caches are not IR objects: identity is retained only by opaque selections.
#[derive(Clone, Default)]
struct PdfStyle {
    font_id: Option<String>,
    font_size: Option<f64>,
    graphic_state: Option<Cursor<types::GraphicState>>,
}
#[derive(Clone, Default)]
struct VisualBbox {
    r#box: Option<std::boxed::Box<Box>>,
}
#[derive(Clone, Default)]
struct PdfCharacter {
    original: Option<Cursor<types::PdfCharacter>>,
    r#box: Option<std::boxed::Box<Box>>,
    visual_bbox: Option<std::boxed::Box<VisualBbox>>,
    pdf_style: Option<std::boxed::Box<PdfStyle>>,
    pdf_character_id: Option<i64>,
    char_unicode: Option<String>,
    scale: Option<f64>,
    vertical: Option<bool>,
    advance: Option<f64>,
    debug_info: Option<bool>,
    xobj_id: Option<i64>,
}
#[derive(Clone, Default)]
struct PdfFormula {
    original: Option<Cursor<types::PdfFormula>>,
    r#box: Option<std::boxed::Box<Box>>,
    pdf_character: Vec<PdfCharacter>,
    pdf_curve: Vec<PdfCurve>,
    pdf_form: Vec<PdfForm>,
    x_offset: Option<f64>,
    y_offset: Option<f64>,
    x_advance: Option<f64>,
}
#[derive(Clone)]
struct PdfCurve {
    original: Cursor<types::PdfCurve>,
    r#box: Option<std::boxed::Box<Box>>,
    relocation_transform: Vec<babeldoc_il_types::PythonValue>,
}
#[derive(Clone)]
struct PdfForm {
    original: Cursor<types::PdfForm>,
    r#box: Option<std::boxed::Box<Box>>,
    relocation_transform: Vec<babeldoc_il_types::PythonValue>,
}
fn read_character(
    c: Cursor<types::PdfCharacter>,
    v: &mut View<'_>,
) -> Result<PdfCharacter, StageError> {
    let style = c
        .pdf_style(v)?
        .map(|s| {
            Ok::<_, StageError>(std::boxed::Box::new(PdfStyle {
                font_id: s.font_id(v)?.clone(),
                font_size: s.font_size(v)?,
                graphic_state: s.graphic_state(v)?.map(|g| g.retain(v)).transpose()?,
            }))
        })
        .transpose()?;
    Ok(PdfCharacter {
        r#box: c
            .r#box(v)?
            .map(|b| read_box(b, v).map(std::boxed::Box::new))
            .transpose()?,
        visual_bbox: c
            .visual_bbox(v)?
            .map(|b| {
                Ok::<_, StageError>(std::boxed::Box::new(VisualBbox {
                    r#box: b
                        .r#box(v)?
                        .map(|b| read_box(b, v).map(std::boxed::Box::new))
                        .transpose()?,
                }))
            })
            .transpose()?,
        pdf_style: style,
        pdf_character_id: c.pdf_character_id(v)?,
        char_unicode: c.char_unicode(v)?.clone(),
        scale: c.scale(v)?,
        vertical: c.vertical(v)?,
        advance: c.advance(v)?,
        debug_info: c.debug_info(v)?,
        xobj_id: c.xobj_id(v)?,
        original: Some(c.retain(v)?),
    })
}
fn read_formula(f: Cursor<types::PdfFormula>, v: &mut View<'_>) -> Result<PdfFormula, StageError> {
    let mut characters = Vec::with_capacity(f.pdf_character_len(v)?);
    for index in 0..f.pdf_character_len(v)? {
        characters.push(read_character(f.pdf_character_at(v, index)?, v)?);
    }
    let mut curves = Vec::with_capacity(f.pdf_curve_len(v)?);
    for index in 0..f.pdf_curve_len(v)? {
        let c = f.pdf_curve_at(v, index)?;
        curves.push(PdfCurve {
            r#box: c
                .r#box(v)?
                .map(|b| read_box(b, v).map(std::boxed::Box::new))
                .transpose()?,
            relocation_transform: c.relocation_transform(v)?.clone(),
            original: c.retain(v)?,
        });
    }
    let mut forms = Vec::with_capacity(f.pdf_form_len(v)?);
    for index in 0..f.pdf_form_len(v)? {
        let c = f.pdf_form_at(v, index)?;
        forms.push(PdfForm {
            r#box: c
                .r#box(v)?
                .map(|b| read_box(b, v).map(std::boxed::Box::new))
                .transpose()?,
            relocation_transform: c.relocation_transform(v)?.clone(),
            original: c.retain(v)?,
        });
    }
    Ok(PdfFormula {
        r#box: f
            .r#box(v)?
            .map(|b| read_box(b, v).map(std::boxed::Box::new))
            .transpose()?,
        pdf_character: characters,
        pdf_curve: curves,
        pdf_form: forms,
        x_offset: f.x_offset(v)?,
        y_offset: f.y_offset(v)?,
        x_advance: f.x_advance(v)?,
        original: Some(f.retain(v)?),
    })
}
fn char_value(
    c: PdfCharacter,
    v: &mut View<'_>,
) -> Result<FieldValue<types::PdfCharacter>, StageError> {
    if let Some(original) = c.original {
        return Ok(original.share(v)?);
    }
    Ok(ir::PdfCharacterInit {
        r#box: c.r#box.map(|b| box_field(*b)),
        visual_bbox: c.visual_bbox.map(|b| {
            ir::VisualBboxInit {
                r#box: b.r#box.map(|b| box_field(*b)),
            }
            .into_value()
        }),
        pdf_style: c
            .pdf_style
            .map(|s| {
                Ok::<_, StageError>(
                    ir::PdfStyleInit {
                        font_id: s.font_id,
                        font_size: s.font_size,
                        graphic_state: s.graphic_state.map(|g| g.share(v)).transpose()?,
                    }
                    .into_value(),
                )
            })
            .transpose()?,
        pdf_character_id: c.pdf_character_id,
        char_unicode: c.char_unicode,
        scale: c.scale,
        vertical: c.vertical,
        advance: c.advance,
        debug_info: c.debug_info,
        xobj_id: c.xobj_id,
        ..Default::default()
    }
    .into_value())
}
fn curve_value(c: PdfCurve, v: &mut View<'_>) -> Result<FieldValue<types::PdfCurve>, StageError> {
    let mut n = c.original.copy_shallow(v)?;
    n.r#box = c.r#box.map(|b| box_field(*b));
    n.relocation_transform = c.relocation_transform;
    Ok(n.into_value())
}
fn form_value(c: PdfForm, v: &mut View<'_>) -> Result<FieldValue<types::PdfForm>, StageError> {
    let mut n = c.original.copy_shallow(v)?;
    n.r#box = c.r#box.map(|b| box_field(*b));
    n.relocation_transform = c.relocation_transform;
    Ok(n.into_value())
}
fn update_formula_data(f: &mut PdfFormula) -> Result<(), StageError> {
    let ignored = |c: &PdfCharacter| {
        c.pdf_character_id.is_none()
            || crate::shared::layout_helper::HEIGHT_NOT_USFUL_CHAR_IN_CHAR
                .iter()
                .any(|s| s.is_some_and(|s| c.char_unicode.as_deref() == Some(s)))
    };
    if f.pdf_character.is_empty() {
        return Err(StageError::failed("min() iterable argument is empty"));
    }
    let eligible = f.pdf_character.iter().any(|c| !ignored(c));
    let (mut x, mut y, mut x2, mut y2) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for c in &f.pdf_character {
        let (a, b, aa, bb) = box_coordinates(need(
            c.visual_bbox.as_deref().and_then(|v| v.r#box.as_deref()),
            "formula character has no visual box",
        )?)?;
        x = x.min(a);
        x2 = x2.max(aa);
        if !eligible || !ignored(c) {
            y = y.min(b);
            y2 = y2.max(bb)
        }
    }
    f.r#box = Some(std::boxed::Box::new(Box {
        x: Some(x),
        y: Some(y),
        x2: Some(x2),
        y2: Some(y2),
    }));
    for a in [&mut f.x_offset, &mut f.y_offset, &mut f.x_advance] {
        if a.unwrap_or_default() == 0. {
            *a = Some(0.)
        }
    }
    Ok(())
}
const LINE_BREAK_PATTERN: &str = r"^[a-zA-Z0-9\x{00c0}-\x{00ff}\x{0100}-\x{017f}\x{0180}-\x{024f}\x{1e00}-\x{1eff}\x{2c60}-\x{2c7f}\x{a720}-\x{a7ff}\x{ab30}-\x{ab6f}\x{0250}-\x{02a0}\x{0400}-\x{04ff}\x{0300}-\x{036f}\x{0500}-\x{052f}\x{0370}-\x{03ff}\x{2de0}-\x{2dff}\x{a650}-\x{a69f}\x{1200}-\x{137f}\x{1380}-\x{139f}\x{2d80}-\x{2ddf}\x{ab00}-\x{ab2f}\x{1e7e0}-\x{1e7ff}\x{0e80}-\x{0eff}\x{0d00}-\x{0d7f}\x{0a80}-\x{0aff}\x{0e00}-\x{0e7f}\x{1000}-\x{109f}\x{aa60}-\x{aa7f}\x{a9e0}-\x{a9ff}\x{116d0}-\x{116ff}\x{0b80}-\x{0bff}\x{0c00}-\x{0c7f}\x{0b00}-\x{0b7f}\x{0530}-\x{058f}\x{10a0}-\x{10ff}\x{1c90}-\x{1cbf}\x{2d00}-\x{2d2f}\x{1780}-\x{17ff}\x{19e0}-\x{19ff}\x{10b00}-\x{10b3f}\x{1d00}-\x{1d7f}\x{1400}-\x{167f}\x{0b00}-\x{0b7f}\x{0780}-\x{07bf}\x{1e900}-\x{1e95f}\x{1c80}-\x{1c8f}\x{1e030}-\x{1e08f}\x{a000}-\x{a48f}\x{a490}-\x{a4cf}'-·ʻ]+$";
#[derive(Clone)]
struct TypesettingUnit {
    pub character: Option<PdfCharacter>,
    pub formula: Option<PdfFormula>,
    pub unicode: Option<String>,
    pub font: Option<Arc<FontFace>>,
    pub original_font: Option<()>,
    pub font_size: Option<f64>,
    pub style: Option<PdfStyle>,
    pub xobj_id: Option<i64>,
    pub debug_info: bool,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub scale: Option<f64>,
    box_cache: Option<Box>,
    width_cache: Option<f64>,
    height_cache: Option<f64>,
    can_break_line_cache: Option<bool>,
    is_cjk_char_cache: Option<bool>,
    mixed_character_blacklist_cache: Option<bool>,
    is_space_cache: Option<bool>,
    is_hung_punctuation_cache: Option<bool>,
    cannot_end_cache: Option<bool>,
    can_passthrough_cache: Option<bool>,
}

impl TypesettingUnit {
    pub fn from_character(character: PdfCharacter, debug_info: bool) -> Self {
        Self::empty(
            Some(character),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            debug_info,
        )
    }
    pub fn from_formula(formula: PdfFormula) -> Self {
        Self::empty(
            None,
            Some(formula),
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
    }
    fn from_unicode(
        unicode: String,
        font: Arc<FontFace>,
        original_font: Option<()>,
        font_size: f64,
        style: PdfStyle,
        xobj_id: Option<i64>,
        debug_info: bool,
    ) -> Self {
        Self::empty(
            None,
            None,
            Some(unicode),
            Some(font),
            original_font,
            Some(font_size),
            Some(style),
            xobj_id,
            debug_info,
        )
    }
    fn empty(
        character: Option<PdfCharacter>,
        formula: Option<PdfFormula>,
        unicode: Option<String>,
        font: Option<Arc<FontFace>>,
        original_font: Option<()>,
        font_size: Option<f64>,
        style: Option<PdfStyle>,
        xobj_id: Option<i64>,
        debug_info: bool,
    ) -> Self {
        Self {
            character,
            formula,
            unicode,
            font,
            original_font,
            font_size,
            style,
            xobj_id,
            debug_info,
            x: None,
            y: None,
            scale: None,
            box_cache: None,
            width_cache: None,
            height_cache: None,
            can_break_line_cache: None,
            is_cjk_char_cache: None,
            mixed_character_blacklist_cache: None,
            is_space_cache: None,
            is_hung_punctuation_cache: None,
            cannot_end_cache: None,
            can_passthrough_cache: None,
        }
    }
    pub fn try_get_unicode(&self) -> Option<&str> {
        if let Some(ch) = &self.character {
            ch.char_unicode.as_deref()
        } else if self.formula.is_some() {
            None
        } else {
            self.unicode.as_deref()
        }
    }
    fn cached<T: Copy>(cache: &mut Option<T>, value: impl FnOnce() -> T) -> T {
        *cache.get_or_insert_with(value)
    }
    pub fn calc_mixed_character_blacklist(&self) -> bool {
        matches!(
            self.try_get_unicode(),
            Some("。" | "，" | "：" | "？" | "！")
        )
    }
    pub fn mixed_character_blacklist(&mut self) -> bool {
        let value = self.calc_mixed_character_blacklist();
        Self::cached(&mut self.mixed_character_blacklist_cache, || value)
    }
    pub fn calc_can_break_line(&self) -> bool {
        let Some(text) = self.try_get_unicode() else {
            return true;
        };
        static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
        !RE.get_or_init(|| Regex::new(LINE_BREAK_PATTERN).expect("literal line-break regex"))
            .is_match(text)
    }
    pub fn can_break_line(&mut self) -> bool {
        let value = self.calc_can_break_line();
        Self::cached(&mut self.can_break_line_cache, || value)
    }
    pub fn calc_is_cjk_char(&self) -> bool {
        if self.formula.is_some() {
            return false;
        }
        let Some(text) = self.try_get_unicode() else {
            return false;
        };
        if text.contains("(cid") || text.chars().count() != 1 {
            return false;
        }
        let ch = text.chars().next().unwrap();
        if "（）【】《》〔〕〈〉〖〗「」『』、。：？！，".contains(ch) {
            return true;
        }
        let cp = ch as u32;
        (0x3000..=0x303f).contains(&cp)
            || (0x3040..=0x309f).contains(&cp)
            || (0x30a0..=0x30ff).contains(&cp)
            || (0x3100..=0x312f).contains(&cp)
            || (0xac00..=0xd7af).contains(&cp)
            || (0x1100..=0x11ff).contains(&cp)
            || (0x3130..=0x318f).contains(&cp)
            || (0xa960..=0xa97f).contains(&cp)
            || (0xd7b0..=0xd7ff).contains(&cp)
            || (0x3190..=0x319f).contains(&cp)
            || (0x3200..=0x32ff).contains(&cp)
            || (0x3300..=0x33ff).contains(&cp)
            || (0x3400..=0x4dbf).contains(&cp)
            || (0x4e00..=0x9fff).contains(&cp)
            || (0x20000..=0x2a6df).contains(&cp)
            || (0x2a700..=0x2b73f).contains(&cp)
            || (0x2b740..=0x2b81f).contains(&cp)
            || (0x2b820..=0x2ceaf).contains(&cp)
            || (0x2ceb0..=0x2ebef).contains(&cp)
            || (0x2ebf0..=0x2ee5f).contains(&cp)
            || (0x30000..=0x3134f).contains(&cp)
            || (0x31350..=0x323af).contains(&cp)
            || (0xff01..=0xff60).contains(&cp)
            || (0xffe0..=0xffe6).contains(&cp)
            || (0x2e80..=0x2eff).contains(&cp)
            || (0x31c0..=0x31ef).contains(&cp)
            || (0x2f00..=0x2fdf).contains(&cp)
            || (0xfe10..=0xfe1f).contains(&cp)
            || (0xfe30..=0xfe4f).contains(&cp)
    }
    pub fn is_cjk_char(&mut self) -> bool {
        let value = self.calc_is_cjk_char();
        Self::cached(&mut self.is_cjk_char_cache, || value)
    }
    pub fn calc_is_space(&self) -> bool {
        self.formula.is_none() && self.try_get_unicode() == Some(" ")
    }
    pub fn is_space(&mut self) -> bool {
        let value = self.calc_is_space();
        Self::cached(&mut self.is_space_cache, || value)
    }
    pub fn calc_is_hung_punctuation(&self) -> bool {
        self.formula.is_none()
            && self.try_get_unicode().is_some_and(|s| {
                matches!(
                    s,
                    "," | "."
                        | ":"
                        | ";"
                        | "?"
                        | "!"
                        | "，"
                        | "。"
                        | "．"
                        | "、"
                        | "："
                        | "；"
                        | "！"
                        | "‼"
                        | "？"
                        | "⁇"
                        | "”"
                        | "’"
                        | "」"
                        | "』"
                        | ")"
                        | "]"
                        | "}"
                        | "）"
                        | "〕"
                        | "〉"
                        | "】"
                        | "〗"
                        | "］"
                        | "｝"
                        | "》"
                        | "～"
                        | "-"
                        | "–"
                        | "—"
                        | "·"
                        | "・"
                        | "‧"
                        | "/"
                        | "／"
                        | "⁄"
                )
            })
    }
    pub fn is_hung_punctuation(&mut self) -> bool {
        let value = self.calc_is_hung_punctuation();
        Self::cached(&mut self.is_hung_punctuation_cache, || value)
    }
    pub fn calc_is_cannot_appear_in_line_end_punctuation(&self) -> bool {
        self.formula.is_none()
            && self.try_get_unicode().is_some_and(|s| {
                matches!(
                    s,
                    "“" | "‘"
                        | "「"
                        | "『"
                        | "("
                        | "["
                        | "{"
                        | "（"
                        | "〔"
                        | "〈"
                        | "《"
                        | "〖"
                        | "〘"
                        | "〚"
                )
            })
    }
    pub fn is_cannot_appear_in_line_end_punctuation(&mut self) -> bool {
        let value = self.calc_is_cannot_appear_in_line_end_punctuation();
        Self::cached(&mut self.cannot_end_cache, || value)
    }
    pub fn calc_can_passthrough(&self) -> bool {
        self.unicode.is_none()
    }
    pub fn can_passthrough(&mut self) -> bool {
        let value = self.calc_can_passthrough();
        Self::cached(&mut self.can_passthrough_cache, || value)
    }
    pub fn box_value(&mut self) -> Result<Box, StageError> {
        if let Some(b) = &self.box_cache {
            return Ok(b.clone());
        }
        let b = if let Some(ch) = &self.character {
            let mut b = ch
                .r#box
                .as_deref()
                .cloned()
                .ok_or_else(|| StageError::Failed("character has no box".into()))?;
            if let Some(v) = ch.visual_bbox.as_deref().and_then(|v| v.r#box.as_deref()) {
                b.y = v.y;
                b.y2 = v.y2;
            }
            b
        } else if let Some(f) = &self.formula {
            f.r#box
                .as_deref()
                .cloned()
                .ok_or_else(|| StageError::Failed("formula has no box".into()))?
        } else {
            let width = self
                .font
                .as_ref()
                .and_then(|f| {
                    f.char_lengths(
                        self.unicode.as_deref().unwrap_or_default(),
                        self.font_size.unwrap_or_default(),
                    )
                    .first()
                    .copied()
                })
                .ok_or_else(|| StageError::Failed("font char_lengths returned no widths".into()))?;
            if self.x.is_none() || self.y.is_none() || self.scale.is_none() {
                Box {
                    x: Some(0.0),
                    y: Some(0.0),
                    x2: Some(width),
                    y2: self.font_size,
                }
            } else {
                let x = self.x.expect("checked");
                let y = self.y.expect("checked");
                Box {
                    x: Some(x),
                    y: Some(y),
                    x2: Some(x + width),
                    y2: Some(y + self.font_size.unwrap_or_default()),
                }
            }
        };
        self.box_cache = Some(b.clone());
        Ok(b)
    }
    pub fn width(&mut self) -> Result<f64, StageError> {
        if let Some(v) = self.width_cache {
            return Ok(v);
        }
        let b = self.box_value()?;
        let (x, _, x2, _) = box_coordinates(&b)?;
        let v = x2 - x;
        self.width_cache = Some(v);
        Ok(v)
    }
    pub fn height(&mut self) -> Result<f64, StageError> {
        if let Some(v) = self.height_cache {
            return Ok(v);
        }
        let b = self.box_value()?;
        let (_, y, _, y2) = box_coordinates(&b)?;
        let v = y2 - y;
        self.height_cache = Some(v);
        Ok(v)
    }
    pub fn try_resue_cache(&mut self, old: &TypesettingUnit) {
        self.is_cjk_char_cache = old.is_cjk_char_cache;
        self.can_break_line_cache = old.can_break_line_cache;
        self.is_space_cache = old.is_space_cache;
        self.is_hung_punctuation_cache = old.is_hung_punctuation_cache;
        self.cannot_end_cache = old.cannot_end_cache;
        self.can_passthrough_cache = old.can_passthrough_cache;
        self.mixed_character_blacklist_cache = old.mixed_character_blacklist_cache;
    }
    pub fn passthrough(&self) -> (Vec<PdfCharacter>, Vec<PdfCurve>, Vec<PdfForm>) {
        if let Some(c) = &self.character {
            (vec![c.clone()], vec![], vec![])
        } else if let Some(f) = &self.formula {
            (
                f.pdf_character.clone(),
                f.pdf_curve.clone(),
                f.pdf_form.clone(),
            )
        } else {
            error!(target: LOG_TARGET, "Cannot passthrough unicode. TypesettingUnit: {self}. ");
            error!(target: LOG_TARGET, "Cannot passthrough unicode. TypesettingUnit: {self}. ");
            (vec![], vec![], vec![])
        }
    }
    pub fn relocate(
        &mut self,
        x: f64,
        y: f64,
        scale: f64,
        debug_info: bool,
    ) -> Result<TypesettingUnit, StageError> {
        if self.character.is_some() {
            let width = self.width()?;
            let height = self.height()?;
            let ch = self.character.as_ref().unwrap();
            let source_style = ch.pdf_style.as_deref().ok_or_else(|| {
                StageError::Failed("'NoneType' object has no attribute 'font_id'".into())
            })?;
            let source_size = source_style.font_size.ok_or_else(|| {
                StageError::Failed(
                    "unsupported operand type(s) for *: 'NoneType' and 'float'".into(),
                )
            })?;
            let new = PdfCharacter {
                pdf_character_id: ch.pdf_character_id,
                char_unicode: ch.char_unicode.clone(),
                r#box: Some(std::boxed::Box::new(Box {
                    x: Some(x),
                    y: Some(y),
                    x2: Some(x + width * scale),
                    y2: Some(y + height * scale),
                })),
                pdf_style: Some(std::boxed::Box::new(PdfStyle {
                    font_id: source_style.font_id.clone(),
                    font_size: Some(source_size * scale),
                    graphic_state: source_style.graphic_state.clone(),
                })),
                scale: Some(scale),
                vertical: ch.vertical,
                advance: ch.advance.filter(|v| *v != 0.0).map(|v| v * scale),
                debug_info: Some(debug_info),
                xobj_id: ch.xobj_id,
                ..Default::default()
            };
            let mut u = TypesettingUnit::from_character(new, debug_info);
            u.try_resue_cache(self);
            Ok(u)
        } else if let Some(f) = &self.formula {
            let fx = f
                .r#box
                .as_deref()
                .and_then(|b| b.x)
                .ok_or_else(|| StageError::Failed("formula box has no x".into()))?;
            let fy = f
                .r#box
                .as_deref()
                .and_then(|b| b.y)
                .ok_or_else(|| StageError::Failed("formula box has no y".into()))?;
            let xo = f.x_offset.ok_or_else(|| {
                StageError::Failed(
                    "unsupported operand type(s) for +: 'float' and 'NoneType'".into(),
                )
            })?;
            let yo = f.y_offset.ok_or_else(|| {
                StageError::Failed(
                    "unsupported operand type(s) for +: 'float' and 'NoneType'".into(),
                )
            })?;
            let x_advance = f.x_advance.ok_or_else(|| {
                StageError::Failed(
                    "unsupported operand type(s) for *: 'NoneType' and 'float'".into(),
                )
            })?;
            let mut chars = Vec::new();
            for ch in &f.pdf_character {
                let b = ch
                    .r#box
                    .as_deref()
                    .ok_or_else(|| StageError::Failed("formula character has no box".into()))?;
                let vb = ch
                    .visual_bbox
                    .as_deref()
                    .and_then(|v| v.r#box.as_deref())
                    .ok_or_else(|| {
                        StageError::Failed("formula character has no visual box".into())
                    })?;
                let (bx, by, bx2, by2) = box_coordinates(b)?;
                let (vx, vy, vx2, vy2) = box_coordinates(vb)?;
                let source_style = ch.pdf_style.as_deref().ok_or_else(|| {
                    StageError::Failed("'NoneType' object has no attribute 'font_id'".into())
                })?;
                let source_size = source_style.font_size.ok_or_else(|| {
                    StageError::Failed(
                        "unsupported operand type(s) for *: 'NoneType' and 'float'".into(),
                    )
                })?;
                let relocated_box = Box {
                    x: Some(x + (bx - fx + xo) * scale),
                    y: Some(y + (by - fy + yo) * scale),
                    x2: Some(x + (bx - fx + (bx2 - bx) + xo) * scale),
                    y2: Some(y + (by - fy + (by2 - by) + yo) * scale),
                };
                let visual_box = Box {
                    x: Some(x + (vx - fx + xo) * scale),
                    y: Some(y + (vy - fy + yo) * scale),
                    x2: Some(x + (vx - fx + (vx2 - vx) + xo) * scale),
                    y2: Some(y + (vy - fy + (vy2 - vy) + yo) * scale),
                };
                let nc = PdfCharacter {
                    pdf_character_id: ch.pdf_character_id,
                    char_unicode: ch.char_unicode.clone(),
                    r#box: Some(std::boxed::Box::new(relocated_box)),
                    visual_bbox: Some(std::boxed::Box::new(VisualBbox {
                        r#box: Some(std::boxed::Box::new(visual_box)),
                    })),
                    pdf_style: Some(std::boxed::Box::new(PdfStyle {
                        font_id: source_style.font_id.clone(),
                        font_size: Some(source_size * scale),
                        graphic_state: source_style.graphic_state.clone(),
                    })),
                    scale: Some(scale),
                    vertical: ch.vertical,
                    advance: ch.advance.filter(|v| *v != 0.0).map(|v| v * scale),
                    xobj_id: ch.xobj_id,
                    ..Default::default()
                };
                chars.push(nc);
            }
            let (minx, miny, maxx, maxy) = bounds(
                chars
                    .iter()
                    .filter_map(|c| c.visual_bbox.as_deref()?.r#box.as_deref()),
            )?;
            let mut nf = PdfFormula {
                r#box: Some(std::boxed::Box::new(Box {
                    x: Some(minx),
                    y: Some(miny),
                    x2: Some(maxx),
                    y2: Some(maxy),
                })),
                pdf_character: chars,
                x_offset: Some(xo * scale),
                y_offset: Some(yo * scale),
                x_advance: Some(x_advance * scale),
                ..Default::default()
            };
            for curve in &f.pdf_curve {
                nf.pdf_curve
                    .push(relocate_curve(curve, fx, fy, x, y, scale, xo, yo)?);
            }
            for form in &f.pdf_form {
                nf.pdf_form
                    .push(relocate_form(form, fx, fy, x, y, scale, xo, yo)?);
            }
            update_formula_data(&mut nf).map_err(|e| StageError::Failed(e.to_string()))?;
            let mut u = TypesettingUnit::from_formula(nf);
            u.try_resue_cache(self);
            Ok(u)
        } else {
            let mut u = TypesettingUnit::from_unicode(
                self.unicode.clone().unwrap_or_default(),
                self.font
                    .clone()
                    .ok_or_else(|| StageError::Failed("unicode unit has no mapped font".into()))?,
                self.original_font.clone(),
                self.font_size.unwrap_or_default() * scale,
                self.style
                    .clone()
                    .ok_or_else(|| StageError::Failed("unicode unit has no style".into()))?,
                self.xobj_id,
                self.debug_info,
            );
            u.x = Some(x);
            u.y = Some(y);
            u.scale = Some(scale);
            u.try_resue_cache(self);
            Ok(u)
        }
    }
    pub fn render(
        &mut self,
    ) -> Result<(Vec<PdfCharacter>, Vec<PdfCurve>, Vec<PdfForm>), StageError> {
        if self.can_passthrough() {
            return Ok(self.passthrough());
        }
        let x = self.x.ok_or_else(|| {
            StageError::Failed("x position must be set, should be set by relocate".into())
        })?;
        let y = self.y.ok_or_else(|| {
            StageError::Failed("y position must be set, should be set by relocate".into())
        })?;
        let scale = self.scale.ok_or_else(|| {
            StageError::Failed("scale must be set, should be set by relocate".into())
        })?;
        let width = self.width()?;
        let size = self.font_size.unwrap_or_default();
        let font = self
            .font
            .as_ref()
            .ok_or_else(|| StageError::Failed("unicode unit has no mapped font".into()))?;
        Ok((
            vec![PdfCharacter {
                pdf_character_id: Some(
                    font.has_glyph(
                        self.unicode
                            .as_deref()
                            .unwrap_or_default()
                            .chars()
                            .next()
                            .unwrap_or('\0') as u32,
                    ) as i64,
                ),
                char_unicode: self.unicode.clone(),
                r#box: Some(std::boxed::Box::new(Box {
                    x: Some(x),
                    y: Some(y),
                    x2: Some(x + width),
                    y2: Some(y + size),
                })),
                pdf_style: Some(std::boxed::Box::new(PdfStyle {
                    font_id: Some(font.id().to_owned()),
                    font_size: Some(size),
                    graphic_state: self.style.as_ref().and_then(|s| s.graphic_state.clone()),
                })),
                scale: Some(scale),
                vertical: Some(false),
                advance: Some(width),
                xobj_id: self.xobj_id,
                debug_info: Some(self.debug_info),
                ..Default::default()
            }],
            vec![],
            vec![],
        ))
    }
}

impl std::fmt::Display for TypesettingUnit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.try_get_unicode().unwrap_or_default())
    }
}

fn bounds<'a>(boxes: impl Iterator<Item = &'a Box>) -> Result<(f64, f64, f64, f64), StageError> {
    let mut b = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    let mut n = 0;
    for x in boxes {
        b.0 = b.0.min(x.x.unwrap_or_default());
        b.1 = b.1.min(x.y.unwrap_or_default());
        b.2 = b.2.max(x.x2.unwrap_or_default());
        b.3 = b.3.max(x.y2.unwrap_or_default());
        n += 1
    }
    if n == 0 {
        Err(StageError::Failed("min() arg is an empty sequence".into()))
    } else {
        Ok(b)
    }
}

fn box_coordinates(box_: &Box) -> Result<(f64, f64, f64, f64), StageError> {
    Ok((
        box_.x
            .ok_or_else(|| StageError::Failed("box.x is None".into()))?,
        box_.y
            .ok_or_else(|| StageError::Failed("box.y is None".into()))?,
        box_.x2
            .ok_or_else(|| StageError::Failed("box.x2 is None".into()))?,
        box_.y2
            .ok_or_else(|| StageError::Failed("box.y2 is None".into()))?,
    ))
}

fn same_box(left: &Box, right: &Box) -> bool {
    left.x == right.x && left.y == right.y && left.x2 == right.x2 && left.y2 == right.y2
}
fn relocation_matrix(tx: f64, ty: f64, s: f64) -> Vec<babeldoc_il_types::PythonValue> {
    [s, 0.0, 0.0, s, tx, ty]
        .into_iter()
        .map(|v| babeldoc_il_types::PythonValue::from(v))
        .collect()
}
fn relocate_curve(
    c: &PdfCurve,
    fx: f64,
    fy: f64,
    x: f64,
    y: f64,
    s: f64,
    xo: f64,
    yo: f64,
) -> Result<PdfCurve, StageError> {
    let mut n = c.clone();
    if let Some(b) = c.r#box.as_deref() {
        n.r#box = Some(std::boxed::Box::new(Box {
            x: Some(x + (b.x.unwrap_or_default() - fx + xo) * s),
            y: Some(y + (b.y.unwrap_or_default() - fy + yo) * s),
            x2: Some(
                x + (b.x.unwrap_or_default() - fx
                    + (b.x2.unwrap_or_default() - b.x.unwrap_or_default())
                    + xo)
                    * s,
            ),
            y2: Some(
                y + (b.y.unwrap_or_default() - fy
                    + (b.y2.unwrap_or_default() - b.y.unwrap_or_default())
                    + yo)
                    * s,
            ),
        }));
    }
    let tx = x + xo * s - fx * s;
    let ty = y + yo * s - fy * s;
    n.relocation_transform = relocation_matrix(tx, ty, s);
    Ok(n)
}
fn relocate_form(
    f: &PdfForm,
    fx: f64,
    fy: f64,
    x: f64,
    y: f64,
    s: f64,
    xo: f64,
    yo: f64,
) -> Result<PdfForm, StageError> {
    let mut n = f.clone();
    if let Some(b) = f.r#box.as_deref() {
        n.r#box = Some(std::boxed::Box::new(Box {
            x: Some(x + (b.x.unwrap_or_default() - fx + xo) * s),
            y: Some(y + (b.y.unwrap_or_default() - fy + yo) * s),
            x2: Some(
                x + (b.x.unwrap_or_default() - fx
                    + (b.x2.unwrap_or_default() - b.x.unwrap_or_default())
                    + xo)
                    * s,
            ),
            y2: Some(
                y + (b.y.unwrap_or_default() - fy
                    + (b.y2.unwrap_or_default() - b.y.unwrap_or_default())
                    + yo)
                    * s,
            ),
        }));
    }
    let tx = x + xo * s - fx * s;
    let ty = y + yo * s - fy * s;
    n.relocation_transform = relocation_matrix(tx, ty, s);
    Ok(n)
}

// Python font selection calls bool() on these runtime bool/int fields.
fn runtime_font_flag(value: &Option<babeldoc_il_types::RuntimeBool>) -> bool {
    match value {
        Some(babeldoc_il_types::RuntimeBool::Bool(value)) => *value,
        Some(babeldoc_il_types::RuntimeBool::Int(value)) => *value != 0,
        None => false,
    }
}
#[derive(Default)]
struct FontTable {
    fonts: HashMap<(Option<i64>, String), SourceFontStyle>,
    xobjects: HashSet<i64>,
}
pub struct Typesetting {
    mapper: Arc<FontMapper>,
    is_cjk: bool,
    progress: bool,
    font_source_styles: HashMap<String, SourceFontStyle>,
}
impl Typesetting {
    pub fn new(
        lang: &str,
        mapper: Arc<FontMapper>,
        progress: bool,
        font_source_styles: HashMap<String, SourceFontStyle>,
    ) -> Self {
        let l = lang.to_uppercase();
        Self {
            mapper,
            is_cjk: ["ZH", "JA", "JP", "KR", "CN", "HK", "TW"]
                .iter()
                .any(|s| l.contains(s)),
            progress,
            font_source_styles,
        }
    }
    fn font_map(&self, page: &Page, v: &View<'_>) -> Result<FontTable, StageError> {
        let read = |f: &Cursor<types::PdfFont>| -> Result<SourceFontStyle, StageError> {
            Ok(SourceFontStyle {
                bold: runtime_font_flag(f.bold(v)?),
                italic: runtime_font_flag(f.italic(v)?),
                monospaced: runtime_font_flag(f.monospace(v)?),
                serif: runtime_font_flag(f.serif(v)?),
            })
        };
        let mut out = FontTable::default();
        let mut fonts = vec![];
        for f in page.pdf_font_iter(v)? {
            let f = f?;
            if let Some(id) = f.font_id(v)?.clone() {
                fonts.push((id, read(&f)?))
            }
        }
        for (id, s) in &fonts {
            out.fonts.insert((None, id.clone()), *s);
        }
        for obj in page.pdf_xobject_iter(v)? {
            let obj = obj?;
            if let Some(id) = obj.xobj_id(v)? {
                out.xobjects.insert(id);
                for (fid, s) in &fonts {
                    out.fonts.insert((Some(id), fid.clone()), *s);
                }
                for f in obj.pdf_font_iter(v)? {
                    let f = f?;
                    if let Some(fid) = f.font_id(v)?.clone() {
                        out.fonts.insert((Some(id), fid), read(&f)?);
                    }
                }
            }
        }
        Ok(out)
    }
    fn create_units(
        &self,
        p: &P,
        fonts: &FontTable,
        v: &mut View<'_>,
    ) -> Result<Vec<TypesettingUnit>, StageError> {
        let mut out = vec![];
        let xobj = p.xobj_id(v)?;
        for child_index in 0..p.pdf_paragraph_composition_len(v)? {
            let comp = p.pdf_paragraph_composition_at(v, child_index)?;
            if let Some(line) = comp.pdf_line(v)? {
                for child_index in 0..line.pdf_character_len(v)? {
                    let c = line.pdf_character_at(v, child_index)?;
                    out.push(TypesettingUnit::from_character(
                        read_character(c, v)?,
                        false,
                    ))
                }
            } else if comp.pdf_character(v)?.is_some() {
                return Err(StageError::failed(
                    "'PdfParagraph' object has no attribute 'debug_info'",
                ));
            } else if let Some(same) = comp.pdf_same_style_characters(v)? {
                for child_index in 0..same.pdf_character_len(v)? {
                    let c = same.pdf_character_at(v, child_index)?;
                    out.push(TypesettingUnit::from_character(
                        read_character(c, v)?,
                        false,
                    ))
                }
            } else if let Some(unis) = comp.pdf_same_style_unicode_characters(v)? {
                let Some(style) = unis.pdf_style(v)? else {
                    continue;
                };
                let Some(id) = style.font_id(v)?.clone() else {
                    continue;
                };
                let asset = xobj
                    .is_none_or(|id| !fonts.xobjects.contains(&id))
                    .then(|| self.mapper.font_by_id(&id))
                    .flatten();
                let sf = if let Some(face) = asset {
                    let t = face.traits();
                    self.font_source_styles
                        .get(&id)
                        // `base` aliases a loaded face; its spelling does not
                        // describe the source font's serif/weight flags.
                        .or_else(|| self.font_source_styles.get(face.id()))
                        .copied()
                        .unwrap_or_else(|| {
                            let lower_id = id.to_lowercase();
                            SourceFontStyle {
                                bold: t.bold,
                                italic: t.italic,
                                monospaced: t.monospaced,
                                serif: lower_id.contains("serif") && !lower_id.contains("sans"),
                            }
                        })
                } else {
                    *fonts
                        .fonts
                        .get(&(xobj.filter(|id| fonts.xobjects.contains(id)), id.clone()))
                        .ok_or_else(|| StageError::failed(format!("font id {id:?} not found")))?
                };
                let size = style.font_size(v)?.unwrap_or_default();
                if size == 0. {
                    return Err(StageError::failed(
                        "Font size must be provided when unicode is provided",
                    ));
                }
                let cached = PdfStyle {
                    font_id: Some(id),
                    font_size: Some(size),
                    graphic_state: style.graphic_state(v)?.map(|g| g.retain(v)).transpose()?,
                };
                for ch in unis
                    .unicode(v)?
                    .as_deref()
                    .unwrap_or_default()
                    .chars()
                    .filter(|c| *c != '\n')
                {
                    if let Some(font) = self.mapper.map(sf, ch) {
                        out.push(TypesettingUnit::from_unicode(
                            ch.to_string(),
                            font,
                            None,
                            size,
                            cached.clone(),
                            xobj,
                            unis.debug_info(v)?.unwrap_or(false),
                        ))
                    }
                }
            } else if let Some(f) = comp.pdf_formula(v)? {
                out.push(TypesettingUnit::from_formula(read_formula(f, v)?))
            }
        }
        Ok(out)
    }
    fn layout(
        &self,
        units: &mut [TypesettingUnit],
        b: &Box,
        scale: f64,
        line_skip: f64,
        first_line_indent: bool,
        english: bool,
    ) -> Result<(Vec<TypesettingUnit>, bool), StageError> {
        let mut sizes = Vec::new();
        for u in units.iter() {
            if let Some(s) = u.font_size.filter(|x| *x != 0.0) {
                sizes.push(s)
            }
            if let Some(s) = u
                .character
                .as_ref()
                .and_then(|c| c.pdf_style.as_deref())
                .and_then(|s| s.font_size)
                .filter(|x| *x != 0.0)
            {
                sizes.push(s)
            }
        }
        sizes.sort_by(f64::total_cmp);
        let dominant = mode(&sizes)?;
        let space = self
            .mapper
            .base_font()
            .char_lengths("你", dominant * scale)
            .first()
            .copied()
            .ok_or_else(|| StageError::Failed("font char_lengths returned no widths".into()))?
            * 0.5;
        sizes.clear();
        for u in units.iter_mut() {
            sizes.push(u.height()?)
        }
        let avg = if sizes.is_empty() {
            0.0
        } else {
            mode(&sizes).unwrap_or_else(|_| sizes.iter().sum::<f64>() / sizes.len() as f64) * scale
        };
        let (bx, by, right, top) = box_coordinates(b)?;
        let mut x = bx;
        let mut y = top - avg;
        let mut line_height: f64 = 0.;
        let mut line_heights = sizes;
        line_heights.clear();
        let mut out: Vec<TypesettingUnit> = Vec::new();
        let mut fit = true;
        if first_line_indent {
            x += space * 4.
        }
        for i in 0..units.len() {
            let w = units[i].width()? * scale;
            let h = units[i].height()? * scale;
            if x == bx && units[i].is_space() {
                continue;
            }
            if let Some(prev) = out.last_mut() {
                let cjk = prev.is_cjk_char() ^ units[i].is_cjk_char();
                let pb = prev.box_value()?;
                let same = if pb.y.is_some_and(|v| v != 0.0) {
                    let previous_y2 = pb
                        .y2
                        .ok_or_else(|| StageError::Failed("box.y2 is None".into()))?;
                    previous_y2 >= y - 0.1 && previous_y2 <= y + line_height + 0.1
                } else {
                    false
                };
                let blacklist =
                    prev.mixed_character_blacklist() || units[i].mixed_character_blacklist();
                if cjk
                    && same
                    && !blacklist
                    && x > bx
                    && units[i].try_get_unicode() != Some(" ")
                    && prev.try_get_unicode() != Some(" ")
                    && !matches!(
                        prev.try_get_unicode(),
                        Some("。" | "！" | "？" | "；" | "：" | "，")
                    )
                {
                    x += space * 0.5
                }
            }
            let before = if english {
                width_before(&mut units[i..], scale)?
            } else {
                0.
            };
            if !units[i].is_hung_punctuation()
                && (x + w > right
                    || (english && x + w + before > right)
                    || (units[i].is_cannot_appear_in_line_end_punctuation() && x + w * 2. > right))
            {
                x = bx;
                if line_heights.is_empty() {
                    return Ok((vec![], false));
                }
                let maxh = line_heights
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let modeh = mode(&line_heights)?;
                y -= (dominant * scale * line_skip)
                    .max(modeh * line_skip)
                    .max(maxh * 1.05);
                line_height = 0.;
                line_heights.clear();
                if y < by {
                    fit = false
                }
                if units[i].is_space() {
                    line_height = line_height.max(h);
                    continue;
                }
            }
            let debug_info = units[i].debug_info;
            let mut relocated = units[i].relocate(x, y, scale, debug_info)?;
            let rb = relocated.box_value()?;
            let previous_x = x;
            x = rb.x2.unwrap_or_default();
            if previous_x > x {
                warn!(target: LOG_TARGET, "坐标回绕！！！TypesettingUnit: {}，", units[i]);
            }
            if !units[i].is_space() {
                line_heights.push(h)
            }
            out.push(relocated);
        }
        Ok((out, fit))
    }
    fn conflict(
        &self,
        b: &Box,
        page: &Page,
        exclude: usize,
        bottom: bool,
        v: &View<'_>,
    ) -> Result<f64, StageError> {
        let crop = read_box(
            need(
                need(page.cropbox(v)?, "page has no cropbox")?.r#box(v)?,
                "cropbox has no box",
            )?,
            v,
        )?;
        let mut limit = if bottom {
            need(crop.y, "cropbox.box.y is None")? * 1.1
        } else {
            need(crop.x2, "cropbox.box.x2 is None")? * 0.9
        };
        let mut boxes = vec![];
        for (i, p) in page.pdf_paragraph_iter(v)?.enumerate() {
            let p = p?;
            if i == exclude {
                continue;
            }
            if let Some(c) = p.r#box(v)? {
                let c = read_box(c, v)?;
                if !same_box(&c, b) {
                    boxes.push(c)
                }
            }
        }
        for c in page.pdf_character_iter(v)? {
            let c = c?;
            boxes.push(read_box(need(c.r#box(v)?, "character has no box")?, v)?)
        }
        for c in page.pdf_figure_iter(v)? {
            let c = c?;
            boxes.push(read_box(need(c.r#box(v)?, "figure has no box")?, v)?)
        }
        let (cx, cy, cx2, cy2) = box_coordinates(b)?;
        for b in boxes {
            let (x, y, x2, y2) = box_coordinates(&b)?;
            if bottom {
                if y2 < cy && x < cx2 && x2 > cx {
                    limit = limit.max(y2)
                }
            } else if x > cx && y < cy2 && y2 > cy {
                limit = limit.min(x)
            }
        }
        Ok(limit)
    }
    fn optimal(
        &self,
        p: &P,
        page: &Page,
        index: usize,
        units: &mut [TypesettingUnit],
        initial: f64,
        english: bool,
        apply: bool,
        v: &mut View<'_>,
    ) -> Result<f64, StageError> {
        let Some(b) = p.r#box(v)? else {
            return Ok(initial);
        };
        let mut b = read_box(b, v)?;
        let mut scale = initial;
        let skip = if self.is_cjk { 1.5 } else { 1.3 };
        let mut expand = 0;
        while scale >= 0.1 {
            if let Ok((placed, fit)) = self.layout(
                units,
                &b,
                scale,
                skip,
                p.first_line_indent(v)?.unwrap_or(false),
                english,
            ) {
                if fit {
                    let mut failed = false;
                    if apply {
                        p.set_scale(v, Some(scale))?;
                        p.set_pdf_paragraph_composition(v, vec![])?;
                        for mut u in placed {
                            match u.render() {
                                Ok((cs, curves, forms)) => {
                                    for c in cs {
                                        let c = char_value(c, v)?;
                                        p.push_pdf_paragraph_composition(
                                            v,
                                            ir::PdfParagraphCompositionInit {
                                                pdf_character: Some(c),
                                                ..Default::default()
                                            }
                                            .into_value(),
                                        )?;
                                    }
                                    for c in curves {
                                        let c = curve_value(c, v)?;
                                        page.push_pdf_curve(v, c)?
                                    }
                                    for f in forms {
                                        let f = form_value(f, v)?;
                                        page.push_pdf_form(v, f)?
                                    }
                                }
                                Err(_) => {
                                    failed = true;
                                    break;
                                }
                            }
                        }
                    }
                    if !failed {
                        return Ok(scale);
                    }
                }
            }
            if p.debug_id(v)?.as_deref().unwrap_or_default().is_empty() {
                return Ok(scale);
            }
            scale -= if scale > 0.6 { 0.05 } else { 0.1 };
            if scale < 0.7 {
                let mut expanded = false;
                if expand == 0 {
                    if let Ok(y) = self.conflict(&b, page, index, true, v) {
                        let y = y + 2.;
                        if y < b.y.unwrap_or_default() {
                            b.y = Some(y);
                            if apply {
                                p.set_box(v, Some(box_field(b.clone())))?
                            }
                            expanded = true
                        }
                    }
                    expand = 1;
                } else if expand == 1 {
                    if let Ok(x) = self.conflict(&b, page, index, false, v) {
                        let x = x - 5.;
                        if x > b.x2.unwrap_or_default() {
                            b.x2 = Some(x);
                            if apply {
                                p.set_box(v, Some(box_field(b.clone())))?
                            }
                            expanded = true
                        }
                    }
                    expand = 2;
                }
                if expanded {
                    continue;
                }
                if expand < 2 {
                    scale = 1.
                }
            }
        }
        if english {
            self.optimal(p, page, index, units, initial, false, apply, v)
        } else {
            Ok(0.1)
        }
    }
    fn render_paragraph(
        &self,
        p: &P,
        page: &Page,
        index: usize,
        fonts: &FontTable,
        v: &mut View<'_>,
    ) -> Result<(), StageError> {
        let mut units = self.create_units(p, fonts, v)?;
        if units.iter_mut().all(TypesettingUnit::can_passthrough) {
            p.set_scale(v, Some(1.))?;
            let mut comps = vec![];
            for u in units {
                if let Some(f) = u.formula {
                    let f = need(f.original, "passthrough formula lost selection")?;
                    comps.push(
                        ir::PdfParagraphCompositionInit {
                            pdf_formula: Some(f.share(v)?),
                            ..Default::default()
                        }
                        .into_value(),
                    );
                } else if let Some(c) = u.character {
                    comps.push(
                        ir::PdfParagraphCompositionInit {
                            pdf_character: Some(char_value(c, v)?),
                            ..Default::default()
                        }
                        .into_value(),
                    );
                }
            }
            p.set_pdf_paragraph_composition(v, comps)?;
            return Ok(());
        }
        let initial = p.optimal_scale(v)?.unwrap_or(1.);
        p.set_pdf_paragraph_composition(v, vec![])?;
        self.optimal(p, page, index, &mut units, initial, true, true, v)?;
        if let Some(order) = p.render_order(v)? {
            let mut sub = 1;
            for child_index in 0..p.pdf_paragraph_composition_len(v)? {
                let comp = p.pdf_paragraph_composition_at(v, child_index)?;
                if let Some(c) = comp.pdf_character(v)? {
                    c.set_render_order(v, Some(order))?;
                    c.set_sub_render_order(v, Some(sub))?;
                    sub += 1
                }
            }
        }
        Ok(())
    }
    fn preprocess(
        &self,
        pages: &[Page],
        v: &mut View<'_>,
        ctx: &mut StageContext<'_>,
    ) -> Result<(), StageError> {
        let mut weighted_scales = HashMap::<u64, (f64, usize)>::new();
        let mut nan_min = None;
        let mut has_weighted_scale = false;
        for page in pages.iter() {
            ctx.report_progress(1, Some(pages.len() as u64 * 2))?;
            let fonts = self.font_map(page, v)?;
            for index in 0..page.pdf_paragraph_len(v)? {
                let p = page.pdf_paragraph_at(v, index)?;
                let result = (|| {
                    let mut units = self.create_units(&p, &fonts, v)?;
                    let count = units
                        .iter()
                        .map(|u| u.formula.as_ref().map_or(1, |f| f.pdf_character.len()))
                        .sum::<usize>();
                    let scale = if units.iter_mut().all(TypesettingUnit::can_passthrough) {
                        1.
                    } else {
                        self.optimal(&p, page, index, &mut units, 1., true, false, v)?
                    };
                    p.set_optimal_scale(v, Some(scale))?;
                    Ok::<_, StageError>(count)
                })();
                let count = match result {
                    Ok(n) => n,
                    Err(e) => {
                        warn!(target:LOG_TARGET,"预处理段落时出错：{e}");
                        p.set_optimal_scale(v, Some(1.))?;
                        0
                    }
                };
                if let Some(scale) = p.optimal_scale(v)? {
                    if count > 0 {
                        has_weighted_scale = true;
                        if scale.is_nan() {
                            nan_min = Some(match nan_min {
                                None => scale,
                                Some(previous) if scale.total_cmp(&previous).is_lt() => scale,
                                Some(previous) => previous,
                            });
                        } else {
                            let key = if scale == 0.0 { 0 } else { scale.to_bits() };
                            let entry = weighted_scales.entry(key).or_insert((scale, 0));
                            entry.1 += count;
                        }
                    }
                }
            }
        }
        if !has_weighted_scale {
            error!(target:LOG_TARGET,"document_scales is empty, there seems no paragraph in this PDF");
        } else {
            // `modes` compared values with `==`: NaNs never matched themselves,
            // so they lose to any non-NaN value and otherwise total-order to the limit.
            let max_count = weighted_scales.values().map(|(_, count)| *count).max();
            let limit = max_count
                .and_then(|max| {
                    weighted_scales
                        .values()
                        .filter(|(_, count)| *count == max)
                        .map(|(scale, _)| *scale)
                        .min_by(f64::total_cmp)
                })
                .or(nan_min)
                .expect("a non-empty weighted scale distribution has a mode");
            for page in pages {
                for child_index in 0..page.pdf_paragraph_len(v)? {
                    let p = page.pdf_paragraph_at(v, child_index)?;
                    if p.optimal_scale(v)?.is_some_and(|s| s > limit) {
                        p.set_optimal_scale(v, Some(limit))?
                    }
                }
            }
        }
        Ok(())
    }
    fn render_page(&self, page: &Page, v: &mut View<'_>) -> Result<(), StageError> {
        let fonts = self.font_map(page, v)?;
        let paragraph_len = page.pdf_paragraph_len(v)?;
        let mut indexed = vec![];
        for i in 0..paragraph_len {
            let p = page.pdf_paragraph_at(v, i)?;
            if let Some(b) = p.r#box(v)? {
                if let Ok(coords) = box_coordinates(&read_box(b, v)?) {
                    indexed.push((i, coords))
                }
            }
        }
        let adjustment = (|| -> Result<(), StageError> {
            for (i, _) in &indexed {
                let Some(b) = page.pdf_paragraph_at(v, *i)?.r#box(v)? else {
                    continue;
                };
                let upper = box_coordinates(&read_box(b.clone(), v)?)?;
                let gap = if upper.3 - upper.1 < 36. { 0.5 } else { 3. };
                let query = (upper.0, upper.1 - gap, upper.2, upper.1);
                let mut max: Option<f64> = None;
                for (j, c) in &indexed {
                    if j == i
                        || !(c.0 <= query.2 && c.2 >= query.0 && c.1 <= query.3 && c.3 >= query.1)
                    {
                        continue;
                    }
                    if let Some(c) = page.pdf_paragraph_at(v, *j)?.r#box(v)? {
                        let lower = box_coordinates(&read_box(c, v)?)?;
                        if !(lower.2 < upper.0 || lower.0 > upper.2) {
                            max = Some(max.map_or(lower.3, |m| m.max(lower.3)))
                        }
                    }
                }
                if let Some(y) = max {
                    let y = y + gap;
                    if y < need(b.y2(v)?, "paragraph.box.y2 is None")? {
                        b.set_y(v, Some(y))?
                    }
                }
            }
            Ok(())
        })();
        if let Err(e) = adjustment {
            warn!(target:LOG_TARGET,"Failed to adjust paragraph positions: {e}")
        }
        for index in 0..paragraph_len {
            let p = page.pdf_paragraph_at(v, index)?;
            self.render_paragraph(&p, page, index, &fonts, v)?
        }
        Ok(())
    }
    pub fn process(
        &self,
        d: &mut ir::Document,
        ctx: &mut StageContext<'_>,
    ) -> Result<(), StageError> {
        let mut v = d.view::<TypesettingView>();
        let pages = v.document().page_iter(&v)?.collect::<Result<Vec<_>, _>>()?;
        if self.progress {
            self.preprocess(&pages, &mut v, ctx)?
        }
        for p in pages.iter() {
            ctx.check_cancelled()?;
            self.render_page(p, &mut v)?;
            if self.progress {
                ctx.report_progress(1, Some(pages.len() as u64 * 2))?;
            }
        }
        drop(pages);
        drop(v);
        d.compact();
        Ok(())
    }
}
fn width_before(units: &mut [TypesettingUnit], scale: f64) -> Result<f64, StageError> {
    if units.is_empty() || units[0].can_break_line() {
        return Ok(0.);
    }
    let mut width = 0.;
    for u in units {
        if u.can_break_line() {
            return Ok(width * scale);
        }
        width += u.width()?
    }
    Ok(width * scale)
}
fn mode(values: &[f64]) -> Result<f64, StageError> {
    if values.is_empty() {
        return Err(StageError::Failed(
            "statistics.mode: no mode for empty data".into(),
        ));
    }
    let mut counts = Vec::<(f64, usize)>::new();
    for v in values {
        if let Some(e) = counts.iter_mut().find(|(x, _)| x == v) {
            e.1 += 1
        } else {
            counts.push((*v, 1))
        }
    }
    let max = counts.iter().map(|e| e.1).max().unwrap();
    Ok(counts.into_iter().find(|e| e.1 == max).unwrap().0)
}
#[cfg(test)]
mod typesetting_tests;
