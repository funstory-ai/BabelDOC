"""Document-scoped Rust handles and LLM callback dispatchers."""

from __future__ import annotations

import json
from concurrent.futures import Future
from typing import Any

from babeldoc.utils.priority_thread_pool_executor import PriorityThreadPoolExecutor


class PriorityServiceDispatcher:
    """Priority queue adapter matching the Rust callback dispatch contract."""

    def __init__(self, service: Any, workers: int):
        self.service = service
        self.record_requests = False
        self.pool = PriorityThreadPoolExecutor(max_workers=max(1, workers))
        self.cancel_flag = None

    def submit(self, method, args, kwargs, completion):
        def invoke():
            if method not in getattr(self.service, "cleanup_methods", frozenset()):
                if self.cancel_flag is not None:
                    self.cancel_flag.checkpoint()
            return getattr(self.service, method)(*args, **(kwargs or {}))

        future: Future = self.pool.submit(invoke)
        future.add_done_callback(completion)

    def close(self):
        self.pool.shutdown(wait=True)


class DocumentRunContext:
    """Python owner for native document handles and two independent dispatchers."""

    def __init__(
        self,
        shared_context,
        font_mapper,
        dispatcher_a,
        dispatcher_b,
    ):
        from babeldoc import _rust_il

        self.shared_context = shared_context
        self.font_mapper = font_mapper
        self.dispatcher_a = dispatcher_a
        self.dispatcher_b = dispatcher_b
        self.font_source_styles = {}
        self.native = _rust_il.DocumentRunContext(
            shared_context,
            font_mapper,
            dispatcher_a,
            dispatcher_b,
        )

    def close(self):
        self.dispatcher_b.close()
        self.dispatcher_a.close()


def _font_assets(config):
    from babeldoc.assets import assets
    from babeldoc.assets.embedding_assets_metadata import get_font_family

    family = get_font_family(config.lang_out)
    paths = {}
    sha3_checksums = {}
    for group in ("normal", "script", "fallback", "base"):
        for font_id in family[group]:
            if font_id in paths:
                continue
            path, metadata = assets.get_font_and_metadata(font_id)
            paths[font_id] = str(path)
            sha3_checksums[font_id] = metadata["sha3_256"]
    return family, paths, sha3_checksums


def create_document_run_context(config):
    """Create the shared Rust state after metadata checks, never at config init."""
    from babeldoc import _rust_il

    family, paths, sha3_checksums = _font_assets(config)
    font_mapper, font_source_styles = _rust_il.font_mapper_handle_with_source_styles(
        json.dumps(family),
        json.dumps(paths),
        json.dumps(sha3_checksums),
        config.primary_font_family,
    )
    shared_context = _rust_il.SharedContextCrossSplitPart(
        list(config.shared_context_cross_split_part.user_glossaries)
    )
    config.shared_context_cross_split_part = shared_context
    # The caller supplies each part's translator after the parser and before
    # entering Rust. The two pools remain independent for LLM-only dispatch.
    dispatcher_a = PriorityServiceDispatcher(None, config.pool_max_workers)
    dispatcher_b = PriorityServiceDispatcher(None, config.pool_max_workers)
    run_context = DocumentRunContext(
        shared_context, font_mapper, dispatcher_a, dispatcher_b
    )
    run_context.font_source_styles = font_source_styles
    run_context.font_family = family
    run_context.font_paths = paths
    return run_context


class TranslatorProbeService:
    """Run the exact high-level LLM probe on the caller's translator object."""

    def __init__(self, translator, probe):
        self.translator = translator
        self.probe_fn = probe

    def probe(self):
        return self.probe_fn(self.translator)


class DebugSerializerService:
    cleanup_methods = frozenset()

    def write_debug_json(self, artifact_json, path):
        from babeldoc.rust_debug_service import RustDebugSerializerService

        return RustDebugSerializerService().write_debug_json(artifact_json, path)
