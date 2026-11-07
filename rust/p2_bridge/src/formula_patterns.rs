//! Lazy Python-re requests at the serial paragraph/style consumer boundary.
use babeldoc_p2_stages::{StageError, shared::formula_helper::FormulaPatternMatcher};
use pyo3::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct PythonFormulaMatcher {
    font_pattern: Option<String>,
    character_pattern: Option<String>,
    cache: Mutex<HashMap<(bool, String), bool>>,
    error: Mutex<Option<PyErr>>,
}
impl std::fmt::Debug for PythonFormulaMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PythonFormulaMatcher (serial only)")
    }
}
impl PythonFormulaMatcher {
    pub fn from_config(config: &Value) -> Arc<Self> {
        let pattern = |key| {
            config
                .get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        };
        let font_pattern = pattern("formular_font_pattern");
        let character_pattern = pattern("formular_char_pattern");
        Arc::new(Self {
            font_pattern,
            character_pattern,
            cache: Mutex::new(HashMap::new()),
            error: Mutex::new(None),
        })
    }
    pub fn take_error(&self) -> Option<PyErr> {
        self.error.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
    fn evaluate(&self, font: bool, text: &str) -> Result<bool, StageError> {
        if font && self.font_pattern.is_none() && !text.starts_with("BASE64:") {
            return babeldoc_p2_stages::shared::formula_helper::is_formulas_font(text, None)
                .map_err(Into::into);
        }
        if !font && self.character_pattern.is_none() {
            return Ok(false);
        }
        let key = (font, text.to_owned());
        if let Some(hit) = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Ok(*hit);
        }
        let result: PyResult<bool> = Python::attach(|py| {
            if font {
                py.import("babeldoc.format.pdf.document_il.utils.formular_helper")?
                    .getattr("is_formulas_font")?
                    .call1((text, self.font_pattern.as_deref()))?
                    .extract()
            } else if let Some(pattern) = self.character_pattern.as_deref() {
                Ok(!py
                    .import("re")?
                    .call_method1("match", (pattern, text))?
                    .is_none())
            } else {
                Ok(false)
            }
        });
        match result {
            Ok(hit) => {
                self.cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(key, hit);
                Ok(hit)
            }
            Err(error) => {
                *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                Err(StageError::Failed("Python formula pattern failed".into()))
            }
        }
    }
}
impl FormulaPatternMatcher for PythonFormulaMatcher {
    fn has_character_pattern(&self) -> bool {
        self.character_pattern.is_some()
    }
    fn font(&self, name: &str) -> Result<bool, StageError> {
        self.evaluate(true, name)
    }
    fn character(&self, text: &str) -> Result<bool, StageError> {
        self.evaluate(false, text)
    }
}
