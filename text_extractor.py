import os

import docx
import PyPDF2
import pytesseract
from PIL import Image, ImageFilter, ImageOps

try:
    import fitz
except ImportError:
    fitz = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _read_docx(file_path):
    doc = docx.Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def _read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _preprocess_image(image):
    grayscale = ImageOps.grayscale(image)
    grayscale = ImageOps.autocontrast(grayscale)
    grayscale = grayscale.filter(ImageFilter.SHARPEN)
    return grayscale


def _ocr_image(image):
    processed = _preprocess_image(image)
    text = pytesseract.image_to_string(processed, config="--psm 6")
    return text.strip()


def _render_pdf_page(page, zoom=2.0):
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for OCR-based PDF extraction.")
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pixmap.n < 4 else "RGBA"
    return Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)


def _read_pdf_text(file_path):
    text_parts = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def _read_pdf_ocr(file_path):
    if fitz is None:
        return ""

    text_parts = []
    document = fitz.open(file_path)
    for page in document:
        image = _render_pdf_page(page)
        page_text = _ocr_image(image)
        if page_text:
            text_parts.append(page_text)
    document.close()
    return "\n".join(text_parts).strip()


def _looks_meaningful(text):
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    alpha_numeric = sum(ch.isalnum() for ch in stripped)
    return alpha_numeric >= 20


def extract_text(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".docx":
        return _read_docx(file_path)

    if extension == ".pdf":
        direct_text = _read_pdf_text(file_path)
        if _looks_meaningful(direct_text):
            return direct_text

        ocr_text = _read_pdf_ocr(file_path)
        if _looks_meaningful(ocr_text):
            if direct_text:
                return f"{direct_text}\n{ocr_text}".strip()
            return ocr_text

        return direct_text

    if extension == ".txt":
        return _read_txt(file_path)

    if extension in IMAGE_EXTENSIONS:
        image = Image.open(file_path)
        return _ocr_image(image)

    return ""