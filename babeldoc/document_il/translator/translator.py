import asyncio
import re
import contextlib
import logging
import threading
import time
import unicodedata
from abc import ABC
from abc import abstractmethod
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import aiohttp
import json
import requests
from loguru import logger
from pydantic_settings import BaseSettings
from dataclasses import dataclass, field
import httpx
import openai
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from babeldoc.document_il.translator.cache import TranslationCache
from babeldoc.document_il.utils.atomic_integer import AtomicInteger


logger = logging.getLogger(__name__)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class RateLimiter:
    def __init__(self, max_qps: int):
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self.last_requests = []  # Track last N requests
        self.window_size = max_qps  # Track requests in a sliding window
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()

            # Clean up old requests outside the 1-second window
            while self.last_requests and now - self.last_requests[0] > 1.0:
                self.last_requests.pop(0)

            # If we have less than max_qps requests in the last second, allow immediately
            if len(self.last_requests) < self.max_qps:
                self.last_requests.append(now)
                return

            # Otherwise, wait until we can make the next request
            next_time = self.last_requests[0] + 1.0
            if next_time > now:
                time.sleep(next_time - now)
            self.last_requests.pop(0)
            self.last_requests.append(next_time)

    def set_max_qps(self, max_qps):
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self.window_size = max_qps


_translate_rate_limiter = RateLimiter(5)


def set_translate_rate_limiter(max_qps):
    _translate_rate_limiter.set_max_qps(max_qps)


class BaseTranslator(ABC):
    # Due to cache limitations, name should be within 20 characters.
    # cache.py: translate_engine = CharField(max_length=20)
    name = "base"
    lang_map = {}

    def __init__(self, lang_in, lang_out, ignore_cache):
        self.ignore_cache = ignore_cache
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
            },
        )

        self.translate_call_count = 0
        self.translate_cache_call_count = 0

    def __del__(self):
        with contextlib.suppress(Exception):
            logger.info(
                f"{self.name} translate call count: {self.translate_call_count}"
            )
            logger.info(
                f"{self.name} translate cache call count: {self.translate_cache_call_count}",
            )

    def add_cache_impact_parameters(self, k: str, v):
        """
        Add parameters that affect the translation quality to distinguish the translation effects under different parameters.
        :param k: key
        :param v: value
        """
        self.cache.add_params(k, v)

    def translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                self.translate_cache_call_count += 1
                return cache
        _translate_rate_limiter.wait()
        translation = self.do_translate(text, rate_limit_params)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(text, translation)
        return translation

    def llm_translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                self.translate_cache_call_count += 1
                return cache
        _translate_rate_limiter.wait()
        translation = self.do_llm_translate(text, rate_limit_params)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(text, translation)
        return translation

    @abstractmethod
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError

    @abstractmethod
    def do_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        logger.critical(
            f"Do not call BaseTranslator.do_translate. "
            f"Translator: {self}. "
            f"Text: {text}. ",
        )
        raise NotImplementedError

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return f"<b{placeholder_id}>"

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return f"</b{placeholder_id}>"

    def get_formular_placeholder(self, placeholder_id: int):
        return self.get_rich_text_left_placeholder(placeholder_id)


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        ignore_cache=False,
    ):
        super().__init__(lang_in, lang_out, ignore_cache)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                )
            ),
        )
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.model = model
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=lambda retry_state: logger.warning(
            f"RateLimitError, retrying in {retry_state.next_action.sleep} seconds... "
            f"(Attempt {retry_state.attempt_number}/100)"
        ),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text),
        )
        self.update_token_count(response)
        return response.choices[0].message.content.strip()

    def prompt(self, text):
        return [
            {
                "role": "system",
                "content": "You are a professional,authentic machine translation engine.",
            },
            {
                "role": "user",
                "content": f";; Treat next line as plain text input and translate it into {self.lang_out}, output translation ONLY. If translation is unnecessary (e.g. proper nouns, codes, {'{{1}}, etc. '}), return the original text. NO explanations. NO notes. Input:\n\n{text}",
            },
        ]

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=lambda retry_state: logger.warning(
            f"RateLimitError, retrying in {retry_state.next_action.sleep} seconds... "
            f"(Attempt {retry_state.attempt_number}/100)"
        ),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        self.update_token_count(response)
        return response.choices[0].message.content.strip()

    def update_token_count(self, response):
        try:
            if response.usage and response.usage.total_tokens:
                self.token_count.inc(response.usage.total_tokens)
            if response.usage and response.usage.prompt_tokens:
                self.prompt_token_count.inc(response.usage.prompt_tokens)
            if response.usage and response.usage.completion_tokens:
                self.completion_token_count.inc(response.usage.completion_tokens)
        except Exception as e:
            logger.exception("Error updating token count")

    def get_formular_placeholder(self, placeholder_id: int):
        return "{v" + str(placeholder_id) + "}", f"{{\\s*v\\s*{placeholder_id}\\s*}}"
        return "{{" + str(placeholder_id) + "}}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return (
            f"<style id='{placeholder_id}'>",
            f"<\\s*style\\s*id\\s*=\\s*'\\s*{placeholder_id}\\s*'\\s*>",
        )

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return "</style>", r"<\s*\/\s*style\s*>"


class TranslationDomain(str, Enum):
    LEGAL = "legal"
    LEGAL_BETA = "legal_beta"
    GENERAL = "general"
    PATENT = "patent"


@dataclass
class TranslationSegment:
    segment: str
    segment_id: int
    paragraph_id: int = 0


@dataclass
class TranslationRequest:
    job_id: int
    source_language: str
    target_language: str
    domain: Union[str, TranslationDomain]
    payload: List[TranslationSegment]

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "domain": (
                self.domain.value
                if isinstance(self.domain, TranslationDomain)
                else self.domain
            ),
            "payload": [vars(segment) for segment in self.payload],
        }


class TranslatorClient:
    def __init__(self, base_url: str = "http://localhost/", port: int = 8005):
        self.base_url = base_url.rstrip("/") + f":{port}/"
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None

    def translate(self, request: TranslationRequest) -> Dict:
        """
        Synchronous translation request
        """
        url = f"{self.base_url}/translate"
        logger.debug(request.to_dict())
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json=request.to_dict()
        )
        response.raise_for_status()
        return response.json()

    async def translate_async(self, request: TranslationRequest) -> Dict:
        """
        Asynchronous translation request
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        url = f"{self.base_url}/translate"
        async with self._session.post(
            url, headers={"Content-Type": "application/json"}, json=request.to_dict()
        ) as response:
            response.raise_for_status()
            return await response.json()


# Usage examples:
"""
# Synchronous usage:
client = TranslatorClient()
request = TranslationRequest(
    job_id=123,
    source_language="en",
    target_language="ko",
    domain=TranslationDomain.LEGAL,
    payload=[
        TranslationSegment(
            segment="Hello, world!",
            segment_id=1
        )
    ]
)
result = client.translate(request)

# Asynchronous usage:
async def main():
    async with TranslatorClient() as client:
        request = TranslationRequest(
            job_id=123,
            source_language="en",
            target_language="ko",
            domain=TranslationDomain.LEGAL,
            payload=[
                TranslationSegment(
                    segment="Hello, world!",
                    segment_id=1
                )
            ]
        )
        result = await client.translate_async(request)

# Multiple async requests:
async def translate_multiple():
    async with TranslatorClient() as client:
        requests = [
            TranslationRequest(
                job_id=i,
                source_language="en",
                target_language="ko",
                domain=TranslationDomain.LEGAL,
                payload=[
                    TranslationSegment(
                        segment=f"Text {i}",
                        segment_id=1
                    )
                ]
            )
            for i in range(3)
        ]
        tasks = [client.translate_async(req) for req in requests]
        results = await asyncio.gather(*tasks)
"""


def segment_generator() -> int:
    """
    Generate a unique segment ID based on the current timestamp.
    Returns an integer representing the current time in milliseconds.
    """
    return int(time.time() * 1000)


# if __name__ == "__main__":
#     client = TranslatorClient()
#     request = TranslationRequest(
#         job_id=123,
#         source_language="en",
#         target_language="ko",
#         domain=TranslationDomain.PATENT,
#         payload=[
#             TranslationSegment(
#                 segment="Hello, world!",
#                 segment_id=1
#             )
#         ]
#     )
#     result = client.translate(request)
class BeringTranslator(BaseTranslator):
    name = "bering"
    CustomPrompt = True

    # lang_in, lang_out, ignore_cache
    def __init__(
        self,
        lang_in=None,
        lang_out=None,
        model=None,
        job_id=None,
        ignore_cache=False,
        rate_limit_params=None,
        *,
        port: int,
        **kwargs,
    ):
        super().__init__(lang_in, lang_out, model, job_id, ignore_cache)
        self.client = TranslatorClient(port=port)

    def validate_translate_args(self, job_id, lang_in, lang_out, domain):
        if not isinstance(job_id, int):
            return False
        if not lang_in or not lang_in.strip():
            return False
        if not lang_out or not lang_out.strip():
            return False
        if not domain or not domain.strip():
            return False
        return True

    def parse_result(self, result: Dict) -> str:
        """
        Parse the translation result from the API response.
        Expected format: {'job_id': int, 'translated_text_segment': [{'segment': str}]}
        """
        if not isinstance(result, dict):
            raise ValueError("Invalid response format: expected dictionary")

        if "translated_text_segment" not in result:
            raise ValueError(
                "Invalid response format: missing 'translated_text_segment'"
            )

        segments = result["translated_text_segment"]
        if not segments or not isinstance(segments, list):
            raise ValueError(
                "Invalid response format: 'translated_text_segment' should be a non-empty list"
            )

        if not segments[0].get("segment"):
            raise ValueError(
                "Invalid response format: missing 'segment' in translated_text_segment"
            )

        return " ".join([segment["segment"] for segment in segments])

    def do_translate(
        self,
        text,
        rate_limit_params: dict = None,
    ) -> str:
        logger.info(f"translate: {text}")
        job_id = self.job_id or 123
        lang_in = self.lang_in
        lang_out = self.lang_out
        domain = self.model
        if not self.validate_translate_args(job_id, lang_in, lang_out, domain):
            raise ValueError("Invalid language or domain")
        request = TranslationRequest(
            job_id=job_id,
            source_language=lang_in,
            target_language=lang_out,
            domain=domain,
            payload=[
                TranslationSegment(segment=text, segment_id=int(segment_generator()))
            ],
        )
        result = self.client.translate(request)
        return self.parse_result(result)

    # def do_llm_translate(self, text, rate_limit_params: dict = None):
    #     logger.info(f"translate: {text}")
    #     job_id = self.job_id or 123
    #     lang_in = self.lang_in
    #     lang_out = self.lang_out
    #     domain = self.model
    #     if not self.validate_translate_args(job_id, lang_in, lang_out, domain):
    #         raise ValueError("Invalid language or domain")
    #     request = TranslationRequest(
    #         job_id=job_id,
    #         source_language=lang_in,
    #         target_language=lang_out,
    #         domain=domain,
    #         payload=[TranslationSegment(segment=text,
    #                                  segment_id=int(segment_generator()))]
    #     )
    #     result = self.client.translate(request)
    #     return self.parse_result(result)
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        raise NotImplementedError(
            "LLM translation is not supported by BeringTranslator"
        )


if __name__ == "__main__":
    translator = BeringTranslator(
        lang_in="en", lang_out="ko", model="legal", job_id=123, ignore_cache=True
    )
    result = translator.translate("Hello, world!")
    print(result)
