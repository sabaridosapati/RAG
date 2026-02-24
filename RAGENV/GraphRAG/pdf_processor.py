"""
PDF processing module using Docling.
Handles PDF extraction and chunking with memory-safe fallbacks.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from PyPDF2 import PdfReader
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


class PDFProcessor:
    """Class for processing PDF documents and extracting chunks."""

    def __init__(self, min_chunk_length: int = 50):
        """
        Initialize PDF processor with memory-conscious Docling configuration.

        Args:
            min_chunk_length: Minimum length of text chunks to extract
        """
        self.min_chunk_length = min_chunk_length

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        pipeline_options.force_backend_text = True
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.generate_table_images = False
        pipeline_options.layout_batch_size = 1
        pipeline_options.ocr_batch_size = 1
        pipeline_options.table_batch_size = 1

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        print("Initialized Docling PDF Processor (memory-safe mode)")

    def extract_chunks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text chunks from a PDF file using Docling.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries containing text chunks and metadata
        """
        print(f"Extracting content from: {pdf_path}")

        try:
            result = self.doc_converter.convert(pdf_path)
            chunks = self._extract_chunks_from_result(result, pdf_path, start_chunk_id=0)
            self._log_chunk_result(chunks)
            return chunks

        except Exception as e:
            message = str(e)
            if self._is_memory_error(message):
                print(f"Memory pressure detected in Docling preprocess: {message}")
                print("Retrying with page-by-page fallback and skipping failed pages...")

                initial_failed_pages = self._parse_failed_pages(message)
                chunks = self._extract_chunks_page_by_page(
                    pdf_path=pdf_path,
                    skip_pages=initial_failed_pages,
                )
                self._log_chunk_result(chunks)
                return chunks

            print(f"Error extracting PDF: {message}")
            raise

    def _extract_chunks_page_by_page(self, pdf_path: str, skip_pages: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """Fallback extractor for problematic documents: convert one page at a time."""
        total_pages = self._get_total_pages(pdf_path)
        skip_pages = set(skip_pages or set())
        chunks: List[Dict[str, Any]] = []
        chunk_id = 0

        print(f"Fallback mode active for {total_pages} pages. Initial skipped pages: {sorted(skip_pages)}")

        for page_num in range(1, total_pages + 1):
            if page_num in skip_pages:
                print(f"Skipping known failing page {page_num}")
                continue

            try:
                result = self.doc_converter.convert(
                    pdf_path,
                    raises_on_error=False,
                    page_range=(page_num, page_num),
                )
            except Exception as e:
                msg = str(e)
                if self._is_memory_error(msg):
                    print(f"Skipping page {page_num} due to memory error: {msg}")
                    skip_pages.add(page_num)
                    continue
                raise

            status = getattr(result, "status", None)
            if str(status).lower().endswith("failure"):
                page_errors = self._extract_error_messages(result)
                if any(self._is_memory_error(err) for err in page_errors):
                    print(f"Skipping page {page_num} after failed conversion (memory-related)")
                    skip_pages.add(page_num)
                    continue
                if page_errors:
                    print(f"Skipping page {page_num} after failed conversion: {' | '.join(page_errors)}")
                    continue

            page_chunks = self._extract_chunks_from_result(
                result=result,
                pdf_path=pdf_path,
                start_chunk_id=chunk_id,
                page_num=page_num,
            )
            chunks.extend(page_chunks)
            chunk_id += len(page_chunks)

        return chunks

    def _extract_chunks_from_result(
        self,
        result,
        pdf_path: str,
        start_chunk_id: int,
        page_num: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Extract chunk dictionaries from a Docling ConversionResult."""
        chunks: List[Dict[str, Any]] = []
        chunk_id = start_chunk_id

        if not getattr(result, "document", None):
            return chunks

        try:
            full_text = result.document.export_to_markdown()
            chunks, chunk_id = self._build_chunks_from_text(
                text=full_text,
                pdf_path=pdf_path,
                start_chunk_id=chunk_id,
                extraction_method="markdown",
                page_num=page_num,
            )
            return chunks

        except Exception as e:
            print(f"Markdown export failed: {e}. Falling back to iterate_items.")

        for element in result.document.iterate_items():
            text = element.text.strip() if hasattr(element, "text") else ""
            if text and len(text) > self.min_chunk_length:
                metadata = {
                    "element_type": element.__class__.__name__,
                    "length": len(text),
                    "extraction_method": "iterate_items",
                }
                if page_num is not None:
                    metadata["page"] = page_num

                chunks.append({
                    "id": f"{Path(pdf_path).stem}_chunk_{chunk_id}",
                    "text": text,
                    "source": Path(pdf_path).name,
                    "chunk_index": chunk_id,
                    "metadata": metadata,
                })
                chunk_id += 1

        return chunks

    def _build_chunks_from_text(
        self,
        text: str,
        pdf_path: str,
        start_chunk_id: int,
        extraction_method: str,
        page_num: Optional[int] = None,
    ):
        """Split extracted text into chunk records."""
        chunks: List[Dict[str, Any]] = []
        chunk_id = start_chunk_id

        paragraphs = text.split("\n\n")
        for para in paragraphs:
            clean_text = para.strip()
            if clean_text and len(clean_text) > self.min_chunk_length:
                metadata: Dict[str, Any] = {
                    "length": len(clean_text),
                    "extraction_method": extraction_method,
                }
                if page_num is not None:
                    metadata["page"] = page_num

                chunks.append({
                    "id": f"{Path(pdf_path).stem}_chunk_{chunk_id}",
                    "text": clean_text,
                    "source": Path(pdf_path).name,
                    "chunk_index": chunk_id,
                    "metadata": metadata,
                })
                chunk_id += 1

        return chunks, chunk_id

    @staticmethod
    def _extract_error_messages(result) -> List[str]:
        errors = getattr(result, "errors", None) or []
        return [str(err) for err in errors]

    @staticmethod
    def _is_memory_error(message: str) -> bool:
        normalized = (message or "").lower()
        return "std::bad_alloc" in normalized or "bad_alloc" in normalized

    @staticmethod
    def _parse_failed_pages(message: str) -> Set[int]:
        """Parse page list from messages like: pages [15, 19]."""
        parsed: Set[int] = set()
        if not message:
            return parsed

        match = re.search(r"pages?\s*\[([^\]]+)\]", message, flags=re.IGNORECASE)
        if not match:
            return parsed

        payload = match.group(1)
        for token in payload.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                parts = token.split("-", 1)
                if parts[0].isdigit() and parts[1].isdigit():
                    start, end = int(parts[0]), int(parts[1])
                    for p in range(min(start, end), max(start, end) + 1):
                        parsed.add(p)
                continue
            if token.isdigit():
                parsed.add(int(token))

        return parsed

    @staticmethod
    def _get_total_pages(pdf_path: str) -> int:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            return len(reader.pages)

    def _log_chunk_result(self, chunks: List[Dict[str, Any]]):
        print(f"Extracted {len(chunks)} chunks from PDF")
        if len(chunks) == 0:
            print("WARNING: No chunks extracted")
            print(f"- Min chunk length: {self.min_chunk_length}")
            print("- Try reducing min_chunk_length in config.py")
            print("- Or check if PDF has extractable text")

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about extracted chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {"total_chunks": 0, "total_chars": 0, "avg_chunk_size": 0}

        total_chars = sum(chunk["metadata"]["length"] for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)

        return {
            "total_chunks": len(chunks),
            "total_chars": total_chars,
            "avg_chunk_size": round(avg_chunk_size, 2),
            "min_chunk_size": min(chunk["metadata"]["length"] for chunk in chunks),
            "max_chunk_size": max(chunk["metadata"]["length"] for chunk in chunks),
        }
