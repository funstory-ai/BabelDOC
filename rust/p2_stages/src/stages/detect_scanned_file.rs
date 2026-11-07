//! 0.6.4-compatible scanned-page thresholding and hit writeback.
use crate::runtime::shared_context::SharedContextRoot;
use crate::{StageContext, StageError};
use babeldoc_ir_core::{self as ir, Cursor, ReadOnlyView, ScanDetectionView, types};

pub trait ScanBackend {
    type Error: From<StageError>;
    fn begin(&mut self) -> Result<(), Self::Error>;
    fn render_before(&mut self, page_number: i64) -> Result<(), Self::Error>;
    fn apply_and_compare(
        &mut self,
        page: &Cursor<types::Page>,
        document: &mut ir::Document,
    ) -> Result<f64, Self::Error>;
    fn close(&mut self) -> Result<(), Self::Error>;
}

/// The narrow contract required from the not-yet-implemented runtime config owner.
pub trait ScanConfig {
    fn auto_enable_ocr_workaround(&self) -> bool;
    fn debug(&self) -> bool;
    fn set_ocr_workaround(&mut self, value: bool);
    fn set_skip_scanned_detection(&mut self, value: bool);
    fn set_disable_rich_text_translate(&mut self, value: bool);
    fn set_remove_non_formula_lines(&mut self, value: bool);
    fn touch_python_config(&mut self) -> Result<(), StageError>;
}

pub trait ScanSharedContext {
    fn set_auto_enabled_ocr_workaround(&mut self, value: bool) -> Result<(), StageError>;
}

impl ScanSharedContext for SharedContextRoot {
    fn set_auto_enabled_ocr_workaround(&mut self, value: bool) -> Result<(), StageError> {
        SharedContextRoot::set_auto_enabled_ocr_workaround(self, value);
        Ok(())
    }
}

pub struct Decision {
    pub checked_pages: Vec<i64>,
    pub scanned: usize,
    pub non_scanned: usize,
    pub hit: bool,
}

pub fn threshold(total: usize) -> (f64, f64) {
    let threshold = (0.8_f64 * total as f64).max(1.0);
    (threshold, total as f64 - threshold)
}

/// Runs the exact count/short-circuit state machine; page numbers are zero-based.
pub fn decide(pages: &[i64], mut is_scanned: impl FnMut(i64) -> bool) -> Decision {
    let (threshold, non_scanned_threshold) = threshold(pages.len());
    let mut scanned = 0usize;
    let mut non_scanned = 0usize;
    let mut checked_pages = Vec::new();
    for page in pages {
        if (scanned as f64) < threshold && (non_scanned as f64) < non_scanned_threshold {
            checked_pages.push(*page);
            if is_scanned(*page) {
                scanned += 1;
            } else {
                non_scanned += 1;
            }
        } else {
            non_scanned += 1;
        }
    }
    Decision {
        checked_pages,
        scanned,
        non_scanned,
        hit: scanned as f64 >= threshold,
    }
}

pub fn process<E, B, C, S>(
    document: &mut ir::Document,
    should_translate: impl Fn(i64) -> bool,
    config: &mut C,
    shared: &mut S,
    backend: &mut B,
    context: &StageContext<'_, E>,
) -> Result<Decision, E>
where
    E: From<StageError>,
    B: ScanBackend<Error = E>,
    C: ScanConfig,
    S: ScanSharedContext,
{
    let pages = {
        let view = document.view::<ReadOnlyView>();
        let selected = view
            .document()
            .page_iter(&view)
            .map_err(|e| E::from(StageError::failed(e.to_string())))?
            .map(|page| {
                let page = page.map_err(|e| E::from(StageError::failed(e.to_string())))?;
                let number = page
                    .page_number(&view)
                    .map_err(|e| E::from(StageError::failed(e.to_string())))?
                    .ok_or_else(|| E::from(StageError::failed("page number missing")))?;
                Ok::<_, E>((number, page))
            })
            .collect::<Result<Vec<_>, E>>()?;
        selected
            .into_iter()
            .filter(|(number, _)| should_translate(*number + 1))
            .collect::<Vec<_>>()
    };
    if pages.is_empty() {
        return Ok(Decision {
            checked_pages: vec![],
            scanned: 0,
            non_scanned: 0,
            hit: false,
        });
    }
    backend.begin()?;
    let total = pages.len();
    let (threshold, non_scanned_threshold) = threshold(total);
    let mut scanned = 0usize;
    let mut non_scanned = 0usize;
    let mut checked_pages = vec![];
    let result = (|| {
        for (number, page) in &pages {
            if (scanned as f64) < threshold && (non_scanned as f64) < non_scanned_threshold {
                backend.render_before(*number)?;
                // Deliberately after render_before and immediately before construction.
                context.check_cancelled()?;
                let similarity = backend.apply_and_compare(page, document)?;
                checked_pages.push(*number);
                if similarity > 0.95 {
                    scanned += 1;
                } else {
                    non_scanned += 1;
                }
            } else {
                non_scanned += 1;
            }
            context.report_progress(1, Some(total as u64))?;
        }
        Ok::<_, E>(())
    })();
    // The Python close request only releases the service's reference; it does not call Document.close().
    let closed = backend.close();
    result?;
    closed?;
    let decision = Decision {
        checked_pages,
        scanned,
        non_scanned,
        hit: scanned as f64 >= threshold,
    };
    if !decision.hit {
        drop(pages);
        document.compact();
        return Ok(decision);
    }
    if config.auto_enable_ocr_workaround() {
        log::warn!(target: "babeldoc.format.pdf.document_il.midend.detect_scanned_file", "Detected {scanned} scanned pages, which is more than 80% of the total pages. Turning on OCR workaround.");
        writeback_on_hit(config, shared, || clean_render_order(document)).map_err(E::from)?;
        drop(pages);
        document.compact();
        Ok(decision)
    } else {
        log::warn!(target: "babeldoc.format.pdf.document_il.midend.detect_scanned_file", "Detected {scanned} scanned pages, which is more than 80% of the total pages. Please check the input PDF file.");
        Err(StageError::ScannedPdf("Scanned PDF detected.".into()).into())
    }
}

fn writeback_on_hit<C, S>(
    config: &mut C,
    shared: &mut S,
    clean: impl FnOnce() -> Result<(), StageError>,
) -> Result<(), StageError>
where
    C: ScanConfig,
    S: ScanSharedContext,
{
    shared.set_auto_enabled_ocr_workaround(true)?;
    config.set_ocr_workaround(true);
    config.set_skip_scanned_detection(true);
    config.set_disable_rich_text_translate(true);
    config.touch_python_config()?;
    clean()?;
    config.set_remove_non_formula_lines(false);
    config.touch_python_config()
}

fn clean_render_order(document: &mut ir::Document) -> Result<(), StageError> {
    let mut view = document.view::<ScanDetectionView>();
    let mut black: Option<Cursor<types::GraphicState>> = None;
    for page_index in 0..view.document().page_len(&view)? {
        let page = view.document().page_at(&view, page_index)?;
        for child_index in 0..page.pdf_character_len(&view)? {
            let character = page.pdf_character_at(&view, child_index)?;
            character.set_render_order(&mut view, None)?;
            if !character.debug_info(&view)?.unwrap_or(false) {
                if let Some(style) = character.pdf_style(&view)? {
                    if let Some(state) = &black {
                        let shared = state.share(&mut view)?;
                        style.set_graphic_state(&mut view, Some(shared))?;
                    } else {
                        style.set_graphic_state(
                            &mut view,
                            Some(
                                ir::GraphicStateInit {
                                    passthrough_per_char_instruction: Some(
                                        ir::Instructions::Materialized("0 g 0 G".into()),
                                    ),
                                    ..Default::default()
                                }
                                .into_value(),
                            ),
                        )?;
                        if let Some(state) = character
                            .pdf_style(&view)?
                            .and_then(|style| style.graphic_state(&view).ok().flatten())
                        {
                            black = Some(state.retain(&mut view)?);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod detect_scanned_file_tests;
