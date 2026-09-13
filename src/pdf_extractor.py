"""
pdf_extractor.py

Utilities to extract raw text from an uploaded PDF resume using pypdf.
"""

from io import BytesIO
from typing import Union

from pypdf import PdfReader


def extract_text_from_pdf(file_obj: Union[BytesIO, str]) -> str:
    """
    Extract plain text from a PDF file.

    Args:
        file_obj: Either a file-like object (e.g. Streamlit's UploadedFile,
                   which behaves like BytesIO) or a path to a PDF file on disk.

    Returns:
        The concatenated text extracted from all pages of the PDF.
        Returns an empty string if no extractable text is found.

    Raises:
        ValueError: If the PDF cannot be read/parsed.
    """
    try:
        reader = PdfReader(file_obj)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read PDF file: {exc}") from exc

    extracted_pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        extracted_pages.append(page_text)

    full_text = "\n".join(extracted_pages).strip()
    return full_text
