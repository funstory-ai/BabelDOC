//! Sequential model iterator driver over E2; CPU fallback never takes the GIL.
use crate::callbacks::{self, CallbackService, CancelFlag, PythonService, Recorder, Request};
use babeldoc_ir_core::Document;
use babeldoc_p2_stages::{ProgressReporter, stages::layout_parser as logic};
use pyo3::{
    exceptions::{PyOverflowError, PyValueError},
    prelude::*,
    types::{PyList, PyTuple},
};
use std::time::Duration;
#[derive(serde::Serialize)]
pub struct LayoutReadings {
    pub model_results: u64,
    pub fallback_start_progress: u64,
    pub total: u64,
}

pub fn inputs(py: Python<'_>, document: &mut Document, characters: bool) -> PyResult<Py<PyAny>> {
    let pages = logic::page_inputs(document, characters)
        .map_err(|e| crate::stage_error::to_python(py, &e))?;
    let out = PyList::empty(py);
    for p in pages {
        let chars = PyList::empty(py);
        if characters {
            for c in p.characters {
                chars.append((c.visual_bbox_present, c.bounds, c.unicode, c.vertical))?;
            }
        }
        out.append((
            p.page_number,
            if characters {
                chars.into_any().unbind()
            } else {
                py.None()
            },
        ))?;
    }
    Ok(out.into_any().unbind())
}
pub fn decode(value: &Bound<'_, PyAny>) -> PyResult<Option<logic::PageResult>> {
    if value.is_none() {
        return Ok(None);
    }
    let tuple = value.cast::<PyTuple>()?;
    let index = tuple.get_item(0)?.extract()?;
    let mediabox = tuple.get_item(1)?.extract::<Option<[f64; 2]>>()?;
    let mut boxes = vec![];
    for b in tuple.get_item(2)?.try_iter()? {
        let b = b?;
        let b = b.cast::<PyTuple>()?;
        let coordinates = b.get_item(0)?.extract::<Vec<(u8, f64)>>()?;
        if coordinates.len() != 4 {
            return Err(PyValueError::new_err("expected four coordinates"));
        }
        let xyxy = std::array::from_fn(|i| {
            if coordinates[i].0 == 32 {
                logic::Coordinate::F32(coordinates[i].1 as f32)
            } else {
                logic::Coordinate::F64(coordinates[i].1)
            }
        });
        if let Some(size) = mediabox {
            logic::convert_box(xyxy, size).map_err(|e| match e {
                logic::ConversionError::Nan => {
                    PyValueError::new_err("cannot convert float NaN to integer")
                }
                logic::ConversionError::Infinite => {
                    PyOverflowError::new_err("cannot convert float infinity to integer")
                }
            })?;
        }
        boxes.push(logic::ModelBox {
            xyxy,
            conf: b.get_item(1)?.extract()?,
            class_name: b.get_item(2)?.extract()?,
        });
    }
    Ok(Some(logic::PageResult {
        index,
        mediabox,
        boxes,
    }))
}
pub fn process(
    py: Python<'_>,
    document: &mut Document,
    dispatcher: Py<PyAny>,
    flag: CancelFlag,
    monitor: Py<PyAny>,
    debug: bool,
    characters: bool,
    dump_path: Option<String>,
) -> PyResult<LayoutReadings> {
    let args = inputs(py, document, characters)?;
    let total = args.bind(py).len()? as u64 * 2;
    let recorder = Recorder::for_dispatcher(py, dispatcher.bind(py));
    let service = PythonService {
        dispatcher,
        recorder,
    };
    callbacks::run(py, 1, |runtime, progress_run| {
        let progress = progress_run
            .start(
                monitor,
                "Parse Page Layout",
                total,
                Duration::from_millis(100),
            )
            .map_err(|e| e.into_python())?;
        let request = |method: &str, job: String, args: Py<PyTuple>| Request {
            stage: "Parse Page Layout".into(),
            job,
            service: "layout".into(),
            method: method.into(),
            args,
            kwargs: None,
        };
        let empty = || Python::attach(|py| PyTuple::empty(py).unbind());
        let body = runtime.block_on(async {
            let begin =
                Python::attach(|py| PyTuple::new(py, [args.clone_ref(py)]).map(Bound::unbind))?;
            service
                .request(request("begin", "begin".into(), begin))
                .await
                .map_err(|e| e.into_python())?;
            let mut sequence = 0;
            loop {
                flag.checkpoint()?;
                let value = service
                    .request(request("next_page", format!("next:{sequence}"), empty()))
                    .await
                    .map_err(|e| e.into_python())?;
                let result = Python::attach(|py| decode(value.bind(py)))?;
                let Some(result) = result else {
                    break;
                };
                logic::apply_model(document, &result)
                    .map_err(|e| Python::attach(|py| crate::stage_error::to_python(py, &e)))?;
                progress.report(1, None).map_err(|e| e.into_python())?;
                sequence += 1;
            }
            let readings = LayoutReadings {
                model_results: sequence,
                fallback_start_progress: progress.completed(),
                total,
            };
            logic::fallback(document, debug, &progress)
                .map_err(|e| Python::attach(|py| crate::stage_error::to_python(py, &e)))?;
            Ok::<_, PyErr>(readings)
        });
        if body.is_err() {
            // Cleanup is explicit and not cancellable at the E2 dispatch boundary.
            let _ = runtime.block_on(service.request(request("close", "close".into(), empty())));
        }
        let readings = match body {
            Ok(readings) => {
                progress.end(None).map_err(|e| e.into_python())?;
                readings
            }
            Err(error) => {
                progress
                    .end(Some(error.into()))
                    .map_err(|e| e.into_python())?;
                unreachable!("stage end propagates primary error")
            }
        };
        runtime
            .block_on(service.request(request("close_process_pool", "pool".into(), empty())))
            .map_err(|e| e.into_python())?;
        if debug {
            if let Some(path) = dump_path {
                let artifact = document
                    .artifact()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                std::fs::write(
                    path,
                    serde_json::to_vec(&artifact)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                )
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
            }
        }
        Ok(readings)
    })
}
