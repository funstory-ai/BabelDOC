import concurrent.futures
import json
import logging
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import Levenshtein
import tiktoken
from tqdm import tqdm

from babeldoc.document_il import Document
from babeldoc.document_il import Page
from babeldoc.document_il import PdfFont
from babeldoc.document_il import PdfParagraph
from babeldoc.document_il.midend.il_translator import DocumentTranslateTracker
from babeldoc.document_il.midend.il_translator import ILTranslator
from babeldoc.document_il.midend.il_translator import PageTranslateTracker
from babeldoc.document_il.translator.translator import BaseTranslator
from babeldoc.document_il.utils.fontmap import FontMapper
from babeldoc.translation_config import TranslationConfig
from babeldoc.document_il.midend.llm_prompt_template import LLMPromptTemplate

enc = tiktoken.encoding_for_model("gpt-4o")

logger = logging.getLogger(__name__)


class BatchParagraph:
    def __init__(
        self, paragraphs: list[PdfParagraph], page_tracker: PageTranslateTracker
    ):
        self.paragraphs = paragraphs
        self.trackers = [page_tracker.new_paragraph() for _ in paragraphs]


class ILTranslatorLLMOnly:
    stage_name = "Translate Paragraphs"

    def __init__(
        self,
        translate_engine: BaseTranslator,
        translation_config: TranslationConfig,
    ):
        self.translate_engine = translate_engine
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)
        self.prompt_template = None
        self.prev_paragraphs = []  # Store previous paragraphs
        self.next_paragraphs = []  # Store following paragraphs
        self.max_context = 3       # Maximum number of context paragraphs
        
        if translation_config.enhanced_prompt and translation_config.prompt_template:
            try:
                self.prompt_template = LLMPromptTemplate.from_xml_file(
                    translation_config.prompt_template,
                    translation_config
                )
                #logger.info(f"Loaded prompt template from {translation_config.prompt_template}")
            except Exception as e:
                logger.error(f"Failed to load prompt template: {e}")

        self.il_translator = ILTranslator(
            translate_engine=translate_engine,
            translation_config=translation_config,
        )

        try:
            self.translate_engine.do_llm_translate(None)
        except NotImplementedError as e:
            raise ValueError("LLM translator not supported") from e

    def find_title_paragraph(self, docs: Document) -> PdfParagraph | None:
        """Find the first paragraph with layout_label 'title' in the document.

        Args:
            docs: The document to search in

        Returns:
            The first title paragraph found, or None if no title paragraph exists
        """
        for page in docs.page:
            for paragraph in page.pdf_paragraph:
                if paragraph.layout_label == "title":
                    logger.info(f"Found title paragraph: {paragraph.unicode}")
                    return paragraph
        return None

    def translate(self, docs: Document) -> None:
        self.translation_config.docs = docs  # Store document reference
        tracker = DocumentTranslateTracker()

        # Try to find the first title paragraph
        title_paragraph = self.find_title_paragraph(docs)

        # count total paragraph
        total = sum(
            [
                len(
                    [
                        p
                        for p in page.pdf_paragraph
                        if p.debug_id is not None and p.unicode is not None
                    ]
                )
                for page in docs.page
            ]
        )
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        ) as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.translation_config.qps,
            ) as executor2:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.translation_config.qps,
                ) as executor:
                    for page in docs.page:
                        self.process_page(
                            page,
                            executor,
                            pbar,
                            tracker.new_page(),
                            title_paragraph,
                            executor2,
                        )

        if self.translation_config.debug:
            path = self.translation_config.get_working_file_path("translate_tracking.json")
            logger.debug(f"save translate tracking to {path}")
            with Path(path).open("w", encoding="utf-8") as f:
                f.write(tracker.to_json())

    def process_page(
        self,
        page: Page,
        executor: concurrent.futures.ThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: PageTranslateTracker = None,
        title_paragraph: PdfParagraph | None = None,
        executor2: concurrent.futures.ThreadPoolExecutor | None = None,
    ):
        self.translation_config.raise_if_cancelled()
        page_font_map = {}
        for font in page.pdf_font:
            page_font_map[font.font_id] = font
        page_xobj_font_map = {}
        for xobj in page.pdf_xobject:
            page_xobj_font_map[xobj.xobj_id] = page_font_map.copy()
            for font in xobj.pdf_font:
                page_xobj_font_map[xobj.xobj_id][font.font_id] = font

        paragraphs = []
        local_title_paragraph = None

        total_unicode_counts = 0
        for paragraph in page.pdf_paragraph:
            if paragraph.debug_id is None or paragraph.unicode is None:
                continue
            # self.translate_paragraph(paragraph, pbar,tracker.new_paragraph(), page_font_map, page_xobj_font_map)
            total_unicode_counts += len(paragraph.unicode)
            paragraphs.append(paragraph)
            if paragraph.layout_label == "title":
                local_title_paragraph = paragraph

            if total_unicode_counts > 1200 or len(paragraphs) > 4:
                self._submit_batch(paragraphs, tracker, executor, pbar, page_font_map,
                                page_xobj_font_map, title_paragraph, local_title_paragraph, executor2)
                paragraphs = []
                total_unicode_counts = 0

        if paragraphs:
            self._submit_batch(paragraphs, tracker, executor, pbar, page_font_map,
                            page_xobj_font_map, title_paragraph, local_title_paragraph, executor2)

    def _submit_batch(self, paragraphs, tracker, executor, pbar, page_font_map,
                    page_xobj_font_map, title_paragraph, local_title_paragraph, executor2):
        """Submit a batch of paragraphs for translation."""
        executor.submit(
            self.translate_paragraph,
            BatchParagraph(paragraphs, tracker),
            pbar,
            page_font_map,
            page_xobj_font_map,
            title_paragraph,
            local_title_paragraph,
            executor2,
        )

    def translate_paragraph(
        self,
        batch_paragraph: BatchParagraph,
        pbar: tqdm | None = None,
        page_font_map: dict[str, PdfFont] = None,
        xobj_font_map: dict[int, dict[str, PdfFont]] = None,
        title_paragraph: PdfParagraph | None = None,
        local_title_paragraph: PdfParagraph | None = None,
        executor=None,
    ):
        """Translate a paragraph using pre and post processing functions."""
        self.translation_config.raise_if_cancelled()
        try:
            inputs = self._prepare_inputs(batch_paragraph, pbar, page_font_map, xobj_font_map)
            if not inputs:
                return

            if self.prompt_template:
                parsed_output = self._translate_with_template(inputs)
            else:
                parsed_output = self._translate_with_default(inputs, title_paragraph, local_title_paragraph)

            self._process_translations(inputs, parsed_output, pbar, page_font_map, xobj_font_map, executor)

        except Exception as e:
            logger.warning(f"Error {e} during translation. try fallback")
            self._fallback_translation(batch_paragraph, pbar, page_font_map, xobj_font_map)

    def _prepare_inputs(self, batch_paragraph, pbar, page_font_map, xobj_font_map):
        """Prepare inputs for translation."""
        inputs = []
        for i in range(len(batch_paragraph.paragraphs)):
            paragraph = batch_paragraph.paragraphs[i]
            tracker = batch_paragraph.trackers[i]
            text, translate_input = self.il_translator.pre_translate_paragraph(
                paragraph, tracker, page_font_map, xobj_font_map
            )
            if text is None:
                pbar.advance(1)
                continue
            inputs.append((text, translate_input, paragraph, tracker))
        return inputs

    def _translate_with_template(self, inputs):
        """Translate using XML template."""
        results = []
        for idx, (text, _, paragraph, _) in enumerate(inputs):
            # Prepare variables for template
            variables = {
                "langFrom": self.translation_config.lang_in,
                "langTo": self.translation_config.lang_out,
                "sourceText": text,
                "contentType": paragraph.layout_label,
                "prevParagraphs": "\n".join(self.prev_paragraphs[-self.max_context:]),
                "nextParagraphs": ""
            }

            # Add context for titles
            if paragraph.layout_label == "title":
                self.next_paragraphs = self._collect_next_paragraphs(paragraph, 3)
                variables["nextParagraphs"] = "\n".join(self.next_paragraphs)

            # Generate template with XML structure
            final_input = self.prompt_template.render(variables)

            # if self.translation_config.debug:
            #     logger.info("="*80)
            #     logger.info(f"Translating {paragraph.layout_label} {idx + 1}/{len(inputs)}")
            #     logger.info("-"*80)
            #     logger.info(f"Source XML:")
            #     logger.info(final_input)
            #     logger.info("-"*80)

            # Get translation from LLM
            llm_output = self.translate_engine.llm_translate(final_input)

            # Process LLM response
            try:
                translation = llm_output.strip()
                
                # if self.translation_config.debug:
                #     logger.info("Raw LLM Response:")
                #     logger.info("-"*80)
                #     logger.info(translation)
                #     logger.info("-"*80)
                
                # Try to parse as XML first
                if "<?xml" in translation or "<TranslationTask" in translation:
                    try:
                        root = ET.fromstring(translation)
                        # Look for translation in the XML structure
                        translation = ""
                        
                        # First try to find direct translation text
                        source_text_elem = root.find(".//SourceText")
                        if source_text_elem is not None and source_text_elem.tail:
                            # Sometimes LLM puts translation right after SourceText
                            translation = source_text_elem.tail.strip()
                        
                        # If not found, concatenate all non-empty text from non-template elements
                        if not translation:
                            for elem in root.iter():
                                # Skip template elements and empty text
                                if (not elem.tag.startswith("if_") and 
                                    elem.text and 
                                    elem.text.strip() and
                                    not any(x in elem.text.lower() for x in ["source", "translate", "context"])):
                                    translation += elem.text.strip() + " "
                        
                        translation = translation.strip()
                        
                        if self.translation_config.debug:
                            logger.info("Extracted from XML:")
                            logger.info(translation)
                    
                    except ET.ParseError as e:
                        logger.warning(f"Failed to parse XML response: {e}")
                        # Use the raw output if XML parsing fails
                        translation = self._clean_response_text(translation)
                else:
                    # Not XML, clean the raw text
                    translation = self._clean_response_text(translation)
                
                results.append({"id": idx, "output": translation})

            except Exception as e:
                logger.error(f"Error processing translation response: {e}")
                # Use raw output as fallback
                results.append({"id": idx, "output": llm_output.strip()})

            # Update context history for non-titles
            if paragraph.layout_label != "title":
                self._update_paragraph_history(text)

            if self.translation_config.debug:
                logger.info("Final Translation:")
                logger.info("-"*80)
                logger.info(results[-1]["output"])
                logger.info("="*80)

        return results

    def _translate_with_default(self, inputs, title_paragraph, local_title_paragraph):
        """Translate using default JSON format."""
        json_format_input = []
        for id_, input_text in enumerate(inputs):
            json_format_input.append({
                "id": id_,
                "input": input_text[0],
                "layout_label": input_text[2].layout_label,
            })

        final_input = self._create_default_prompt(
            json.dumps(json_format_input, ensure_ascii=False, indent=2),
            title_paragraph,
            local_title_paragraph
        )

        # if self.translation_config.debug:
        #     logger.info("="*80)
        #     logger.info("Using Default Template")
        #     logger.info("-"*80)
        #     logger.info(final_input)
        #     logger.info("="*80)

        llm_output = self.translate_engine.llm_translate(final_input)
        llm_output = self._clean_json_output(llm_output.strip())
        return json.loads(llm_output)

    def _create_default_prompt(self, json_input, title_paragraph, local_title_paragraph):
        """Create default JSON-based prompt."""
        llm_input = ["You are a professional, authentic machine translation engine."]

        if title_paragraph:
            llm_input.append(f"The first title in the full text: {title_paragraph.unicode}")
        if (local_title_paragraph and 
            local_title_paragraph.debug_id != title_paragraph.debug_id):
            llm_input.append(
                f"The most similar title in the full text: {local_title_paragraph.unicode}"
            )
        # Create a structured prompt template for LLM translation
        prompt_template = f"""
        You will be given a JSON formatted input containing entries with "id" and "input" fields.
        Here is the input:
        
        ```json
        {json_input}
        ```
        
        For each entry in the JSON, translate the contents of the "input" field into {self.translation_config.lang_out}.
        Write the translation back into the "output" field for that entry.
        
        Here is an example of the expected format:
        <example>
        ```json
        Input:
        {{
            "id": 1,
            "input": "Source",
            "layout_label": "plain text"
        }}
        ```
        Output:
        ```json
        {{
            "id": 1,
            "output": "Translation"
        }}
        ```
        </example>
        
        Please return the translated json directly without wrapping ```json``` tag or include any additional information.
        """
        llm_input.append(prompt_template)
        return "\n".join(llm_input).strip()

    def _process_translations(self, inputs, parsed_output, pbar, page_font_map, xobj_font_map, executor):
        """Process translation results."""
        translation_results = {item["id"]: item["output"] for item in parsed_output}

        if len(translation_results) != len(inputs):
            raise Exception(
                f"Translation results length mismatch. Expected: {len(inputs)}, Got: {len(translation_results)}"
            )

        for id_, output in translation_results.items():
            self._process_single_translation(
                id_, output, inputs, pbar, page_font_map, xobj_font_map, executor
            )

    def _process_single_translation(self, id_, output, inputs, pbar, page_font_map, xobj_font_map, executor):
        """Process a single translation result."""
        should_fallback = True
        try:
            if not isinstance(output, str):
                logger.warning(f"Translation result is not a string. Output: {output}")
                return

            id_ = int(id_)
            if id_ >= len(inputs):
                logger.warning(f"Invalid id {id_}, skipping")
                return

            translated_text = re.sub(r"[. 。…，]{20,}", ".", output)
            translate_input = inputs[id_][1]
            input_unicode = inputs[id_][2].unicode
            output_unicode = translated_text

            if not self._validate_translation(input_unicode, output_unicode):
                return

            self.il_translator.post_translate_paragraph(
                inputs[id_][2],
                inputs[id_][3],
                translate_input,
                translated_text,
            )
            if pbar:
                pbar.advance(1)
            should_fallback = False

        except Exception as e:
            logger.exception(f"Error translating paragraph. Error: {e}.")

        finally:
            if should_fallback:
                self._handle_fallback(inputs[id_], pbar, page_font_map, xobj_font_map, executor)

    def _validate_translation(self, input_unicode: str, output_unicode: str) -> bool:
        """Validate translation output."""
        input_token_count = len(enc.encode(input_unicode))
        output_token_count = len(enc.encode(output_unicode))

        if not (0.3 < output_token_count / input_token_count < 3):
            logger.warning(
                f"Translation result is too long or too short. Input: {input_token_count}, Output: {output_token_count}"
            )
            return False

        edit_distance = Levenshtein.distance(input_unicode, output_unicode)
        if edit_distance < 5 and input_token_count > 20:
            logger.warning(
                f"Translation result edit distance is too small. distance: {edit_distance}, input: {input_unicode}, output: {output_unicode}"
            )
            return False

        return True

    def _handle_fallback(self, input_data, pbar, page_font_map, xobj_font_map, executor):
        """Handle translation fallback."""
        logger.warning(f"Fallback to simple translation. paragraph id: {input_data[2].debug_id}")
        executor.submit(
            self.il_translator.translate_paragraph,
            input_data[2],
            pbar,
            input_data[3],
            page_font_map,
            xobj_font_map,
        )

    def _fallback_translation(self, batch_paragraph, pbar, page_font_map, xobj_font_map):
        """Process fallback translation for entire batch."""
        for i in range(len(batch_paragraph.paragraphs)):
            paragraph = batch_paragraph.paragraphs[i]
            tracker = batch_paragraph.trackers[i]
            if paragraph.debug_id is None:
                continue
            self.il_translator.translate_paragraph(
                paragraph,
                pbar,
                tracker,
                page_font_map,
                xobj_font_map,
            )

    def _collect_next_paragraphs(self, current_paragraph: PdfParagraph, count: int) -> list[str]:
        """Collect the next few paragraphs after the current one."""
        collected = []
        found_current = False
        
        for page in self.translation_config.docs.page:
            for paragraph in page.pdf_paragraph:
                if paragraph == current_paragraph:
                    found_current = True
                    continue
                    
                if found_current and paragraph.unicode:
                    if paragraph.layout_label == "title":
                        break
                    collected.append(paragraph.unicode)
                    if len(collected) >= count:
                        break
                        
            if len(collected) >= count:
                break
                
        return collected

    def _update_paragraph_history(self, text: str):
        """Update the paragraph history with new text."""
        if text and isinstance(text, str):
            self.prev_paragraphs.append(text)
            if len(self.prev_paragraphs) > self.max_context:
                self.prev_paragraphs.pop(0)

    def _clean_response_text(self, text: str) -> str:
        """Clean up response text by removing unwanted elements.
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned translation text
        """
        # Remove common wrappers
        for wrapper in ["<json>", "</json>", "```json", "```", "Translation:", "Output:"]:
            text = text.replace(wrapper, "").strip()
            
        # Remove XML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
