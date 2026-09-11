from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

from babeldoc import __version__ as babeldoc_version
from babeldoc.format.pdf.translation_config import TranslateResult
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode
from babeldoc.glossary import Glossary
from babeldoc.progress_monitor import ProgressMonitor
from babeldoc.tools.executor.translator import ExecutorTranslator
from babeldoc.tools.executor.workroot import get_workroot
from babeldoc.tools.executor.workroot import relative_to_workroot
from babeldoc.tools.executor.workroot import resolve_dir
from babeldoc.tools.executor.workroot import resolve_file
from babeldoc.translator.translator import set_translate_rate_limiter

logger = logging.getLogger(__name__)


def run_babeldoc_request(request: dict[str, Any], progress_send, cancel_recv) -> None:
    cancel_state = _CancelState()
    stop_cancel_watcher: threading.Event | None = None
    cancel_watcher: threading.Thread | None = None
    task_id = _task_id(request)

    def emit(event_type: str, payload: dict[str, Any]) -> None:
        progress_send.send({"type": event_type, "payload": payload})

    try:
        config_started_at = time.monotonic()
        config = build_translation_config(request, task_id=task_id)
        config_elapsed = time.monotonic() - config_started_at
        logger.info(
            "BabelDOC execution started: task_id=%s input=%s output_dir=%s pages=%s lang_in=%s lang_out=%s model=%s config_elapsed=%.3fs",
            task_id,
            getattr(config, "input_file", None),
            getattr(config, "output_dir", None),
            getattr(config, "pages", None),
            getattr(config, "lang_in", None),
            getattr(config, "lang_out", None),
            getattr(config, "model", None),
            config_elapsed,
        )
        cancel_state.attach_config(config)
        stop_cancel_watcher = threading.Event()
        cancel_watcher = threading.Thread(
            target=_watch_cancel_pipe,
            args=(cancel_recv, cancel_state, stop_cancel_watcher),
            name="executor-cancel-watch",
            daemon=True,
        )
        cancel_watcher.start()

        result = _run_async_translate(config, emit)
        logger.info("BabelDOC execution finished: task_id=%s", task_id)
        emit("result", translate_result_to_payload(result, config))
    except Exception as exc:
        logger.exception("BabelDOC execution failed: task_id=%s", task_id)
        user_message = (
            exc.message_for_user if isinstance(exc, BabelDocReportedError) else None
        )
        emit(
            "error",
            {
                "code": _error_code(exc),
                "message": _safe_message(exc),
                "message_for_user": user_message,
                "details": {"exception_type": exc.__class__.__name__},
            },
        )
    finally:
        if stop_cancel_watcher is not None:
            stop_cancel_watcher.set()
        progress_send.send(None)
        if cancel_watcher is not None:
            cancel_watcher.join(timeout=1)


class _CancelState:
    def __init__(self):
        self._lock = threading.Lock()
        self._config: TranslationConfig | None = None
        self._cancel_requested = False

    def request_cancel(self) -> None:
        with self._lock:
            self._cancel_requested = True
            config = self._config
        if config is not None:
            config.cancel_translation()

    def attach_config(self, config: TranslationConfig) -> None:
        with self._lock:
            self._config = config
            cancel_requested = self._cancel_requested
        if cancel_requested:
            config.cancel_translation()


def _watch_cancel_pipe(cancel_recv, cancel_state: _CancelState, stop_event) -> None:
    while not stop_event.is_set():
        try:
            if cancel_recv.poll(0.05):
                cancel_recv.recv()
                cancel_state.request_cancel()
                return
        except (BrokenPipeError, EOFError, OSError):
            return


def build_translation_config(
    request: dict[str, Any],
    *,
    task_id: str | None = None,
) -> TranslationConfig:
    total_started_at = time.monotonic()
    timing: dict[str, float] = {}

    def mark(name: str, started_at: float) -> None:
        timing[name] = time.monotonic() - started_at

    workroot = get_workroot()
    paths = _required_object(request, "paths")
    translation = _required_object(request, "translation_config")
    runtime_limits = _required_object(request, "runtime_limits")
    gateways = _required_object(request, "gateways")
    assets = _optional_object(request, "assets")
    metadata = _optional_object(request, "metadata")

    started_at = time.monotonic()
    input_file = resolve_file(workroot, _required_str(paths, "input_file"))
    output_dir = resolve_dir(workroot, _required_str(paths, "output_dir"), create=True)
    working_dir = resolve_dir(
        workroot,
        _optional_str(paths, "working_dir") or _required_str(paths, "output_dir"),
        create=True,
    )
    mark("paths", started_at)

    started_at = time.monotonic()
    qps = _required_int(runtime_limits, "qps")
    report_interval = _required_number(runtime_limits, "report_interval_seconds")
    set_translate_rate_limiter(qps)

    max_pages_per_part = _required_int(runtime_limits, "max_pages_per_part")
    split_strategy = TranslationConfig.create_max_pages_per_part_split_strategy(
        max_pages_per_part
    )
    mark("limits", started_at)

    started_at = time.monotonic()
    translator = _create_translator(
        _required_object(gateways, "main_llm"),
        translation,
    )
    mark("main_translator", started_at)

    started_at = time.monotonic()
    term_translator = _create_translator(
        _required_object(gateways, "ate_llm"),
        translation,
    )
    mark("term_translator", started_at)

    started_at = time.monotonic()
    doc_layout_model = _create_doc_layout_model(_required_object(gateways, "layout"))
    mark("layout_model", started_at)

    primary_font_family = _optional_str(translation, "primary_font_family")
    if primary_font_family == "none":
        primary_font_family = None

    started_at = time.monotonic()
    glossaries = _load_glossaries(
        workroot,
        assets,
        _required_str(translation, "lang_out"),
    )
    mark("glossaries", started_at)

    started_at = time.monotonic()
    config = TranslationConfig(
        input_file=str(input_file),
        output_dir=str(output_dir),
        working_dir=str(working_dir),
        translator=translator,
        term_extraction_translator=term_translator,
        debug=_required_bool(translation, "debug"),
        lang_in=_required_str(translation, "lang_in"),
        lang_out=_required_str(translation, "lang_out"),
        pages=_optional_str(translation, "pages"),
        no_dual=_required_bool(translation, "no_dual"),
        no_mono=_required_bool(translation, "no_mono"),
        qps=qps,
        doc_layout_model=doc_layout_model,
        skip_clean=_required_bool(translation, "skip_clean"),
        dual_translate_first=_required_bool(translation, "dual_translate_first"),
        disable_rich_text_translate=_required_bool(
            translation, "disable_rich_text_translate"
        ),
        enhance_compatibility=False,
        use_side_by_side_dual=_required_bool(translation, "use_side_by_side_dual"),
        use_alternating_pages_dual=_required_bool(
            translation, "use_alternating_pages_dual"
        ),
        report_interval=report_interval,
        progress_monitor=ProgressMonitor(
            [("translate", 1.0)],
            report_interval=report_interval,
        ),
        watermark_output_mode=WatermarkOutputMode.NoWatermark,
        split_strategy=split_strategy,
        skip_scanned_detection=_required_bool(translation, "skip_scanned_detection"),
        ocr_workaround=_required_bool(translation, "ocr_workaround"),
        custom_system_prompt=_optional_str(translation, "custom_system_prompt"),
        glossaries=glossaries,
        pool_max_workers=_required_int(runtime_limits, "pool_max_workers"),
        auto_extract_glossary=_required_bool(translation, "auto_extract_glossary"),
        auto_enable_ocr_workaround=_required_bool(
            translation, "auto_enable_ocr_workaround"
        ),
        primary_font_family=primary_font_family,
        only_include_translated_page=_required_bool(
            translation, "only_include_translated_page"
        ),
        save_auto_extracted_glossary=True,
        merge_alternating_line_numbers=_required_bool(
            translation, "merge_alternating_line_numbers"
        ),
        remove_non_formula_lines=_required_bool(
            translation, "remove_non_formula_lines"
        ),
        metadata_extra_data=_optional_str(metadata, "metadata_extra_data"),
        term_pool_max_workers=_required_int(runtime_limits, "term_pool_max_workers"),
        enable_translation_tracking=_optional_bool(
            translation,
            "enable_translation_tracking",
        ),
    )
    mark("translation_config", started_at)

    started_at = time.monotonic()
    getattr(doc_layout_model, "init_font_mapper", lambda _config: None)(config)
    mark("font_mapper", started_at)

    total_elapsed = time.monotonic() - total_started_at
    logger.info(
        "BabelDOC config timing: task_id=%s total=%.3fs paths=%.3fs limits=%.3fs main_translator=%.3fs term_translator=%.3fs layout_model=%.3fs glossaries=%.3fs translation_config=%.3fs font_mapper=%.3fs glossary_count=%s",
        task_id or _task_id(request),
        total_elapsed,
        timing.get("paths", 0.0),
        timing.get("limits", 0.0),
        timing.get("main_translator", 0.0),
        timing.get("term_translator", 0.0),
        timing.get("layout_model", 0.0),
        timing.get("glossaries", 0.0),
        timing.get("translation_config", 0.0),
        timing.get("font_mapper", 0.0),
        len(glossaries),
    )
    return config


def _task_id(request: dict[str, Any]) -> str:
    value = request.get("task_id")
    return value if isinstance(value, str) and value else "unknown"


def translate_result_to_payload(
    result: TranslateResult, config: TranslationConfig
) -> dict[str, Any]:
    workroot = get_workroot()
    files = {
        "mono_pdf": relative_to_workroot(workroot, result.mono_pdf_path),
        "dual_pdf": relative_to_workroot(workroot, result.dual_pdf_path),
        "mono_no_watermark_pdf": relative_to_workroot(
            workroot, result.no_watermark_mono_pdf_path
        ),
        "dual_no_watermark_pdf": relative_to_workroot(
            workroot, result.no_watermark_dual_pdf_path
        ),
        "auto_extracted_glossary_csv": relative_to_workroot(
            workroot, result.auto_extracted_glossary_path
        ),
    }
    payload = {
        "files": {key: value for key, value in files.items() if value},
        "metrics": {
            "time_consume_seconds": _number_or_zero(
                getattr(result, "total_seconds", None)
            ),
            "peak_memory_usage": _number_or_zero(
                getattr(result, "peak_memory_usage", None)
            ),
            "pdf_total_char_count": int(
                getattr(result, "total_valid_character_count", 0) or 0
            ),
            "pdf_total_char_token_count": int(
                getattr(result, "total_valid_text_token_count", 0) or 0
            ),
        },
        "pages": _pages_to_string(config),
    }
    if config.enable_translation_tracking:
        payload["translation_tracking"] = result.translation_tracking
    return payload


def _run_async_translate(config: TranslationConfig, emit) -> TranslateResult:
    from babeldoc.format.pdf import high_level

    emit(
        "progress",
        {
            "type": "babeldoc_version",
            "version": babeldoc_version,
        },
    )

    async def run() -> TranslateResult:
        async for event in high_level.async_translate(config):
            event_type = event.get("type")
            if event_type == "finish":
                result = event.get("translate_result")
                if isinstance(result, TranslateResult):
                    return result
                raise ValueError("BabelDOC finish event did not contain result")
            if event_type == "error":
                raise BabelDocReportedError(
                    str(event.get("error")),
                    event.get("message_for_user"),
                )
            emit("progress", dict(event))
        raise RuntimeError("BabelDOC async_translate ended without finish event")

    return asyncio.run(run())


class BabelDocReportedError(RuntimeError):
    def __init__(self, message: str, message_for_user: Any):
        super().__init__(message)
        self.message_for_user = (
            message_for_user if isinstance(message_for_user, str) else None
        )


def _create_translator(gateway: dict[str, Any], translation: dict[str, Any]):
    return ExecutorTranslator(
        lang_in=_required_str(translation, "lang_in"),
        lang_out=_required_str(translation, "lang_out"),
        model=_required_str(gateway, "model"),
        base_url=_required_str(gateway, "base_url"),
        api_key=_required_str(gateway, "api_key"),
    )


def _create_doc_layout_model(layout: dict[str, Any]):
    adapter = _required_str(layout, "adapter")
    if adapter != "rpc_doclayout8":
        raise ValueError("gateways.layout.adapter must be rpc_doclayout8")

    from babeldoc.docvision.rpc_doclayout8 import RpcDocLayoutModel

    return RpcDocLayoutModel(
        host=_required_str(layout, "base_url"),
        requires_line_extraction=_required_bool(layout, "requires_line_extraction"),
    )


def _load_glossaries(
    workroot: Path,
    assets: dict[str, Any],
    lang_out: str,
) -> list[Glossary]:
    glossaries = assets.get("glossaries") or []
    if not isinstance(glossaries, list):
        raise ValueError("assets.glossaries must be an array")
    loaded: list[Glossary] = []
    for item in glossaries:
        if not isinstance(item, dict):
            raise ValueError("glossary asset must be an object")
        path = resolve_file(workroot, _required_str(item, "path"))
        glossary = Glossary.from_csv(path, lang_out)
        name = _optional_str(item, "name")
        if name is not None:
            glossary.name = name
        loaded.append(glossary)
    return loaded


def _pages_to_string(config: TranslationConfig) -> str | None:
    pages = getattr(config, "pages", None)
    if pages:
        return str(pages)
    page_ranges = getattr(config, "page_ranges", None)
    if not page_ranges:
        return None
    return ",".join(f"{start}-{end}" for start, end in page_ranges)


def _error_code(exc: Exception) -> str:
    name = exc.__class__.__name__.lower()
    if "scanned" in name:
        return "babeldoc_scanned_pdf"
    if isinstance(exc, TimeoutError):
        return "subprocess_timeout"
    if isinstance(exc, FileNotFoundError | ValueError):
        return "invalid_output"
    return "babeldoc_failed"


def _safe_message(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        return "translation failed"
    return text.splitlines()[0][:500]


def _number_or_zero(value: Any) -> float | int:
    if isinstance(value, int | float):
        return value
    return 0


def _required_object(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _optional_object(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _required_str(root: dict[str, Any], key: str) -> str:
    value = root.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(root: dict[str, Any], key: str) -> str | None:
    value = root.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _required_bool(root: dict[str, Any], key: str) -> bool:
    value = root.get(key)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean")


def _optional_bool(root: dict[str, Any], key: str, default: bool = False) -> bool:
    value = root.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean")


def _required_int(root: dict[str, Any], key: str) -> int:
    value = root.get(key)
    if isinstance(value, int):
        return value
    raise ValueError(f"{key} must be an integer")


def _required_number(root: dict[str, Any], key: str) -> float:
    value = root.get(key)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"{key} must be a number")
