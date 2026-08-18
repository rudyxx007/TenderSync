import os
from typing import Optional
from docling.document_converter import DocumentConverter

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}


def is_supported_document(filename: Optional[str]) -> bool:
    """
    Check whether the given filename has a supported document extension.
    """
    if not filename:
        return False
    ext = os.path.splitext(filename.lower())[1]
    return ext in SUPPORTED_EXTENSIONS


def extract_document_text(file_path: str, filename: str) -> str:
    """
    Extract text and layout from PDF or Image:
    1. Standalone Images (.png, .jpg, .jpeg, .webp, .tiff, .bmp):
       - Uses RapidOCR on the image array directly.
    2. PDF Documents (.pdf):
       - First tries vector text extraction (fast for digital PDFs) using pypdfium2.
       - If scanned / sparse text, performs RapidOCR across rendered page images.
       - Fallback to DocumentConverter.
    """
    ext = os.path.splitext(filename.lower())[1]

    # Handle image formats
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
        try:
            from rapidocr import RapidOCR
            import numpy as np
            from PIL import Image

            ocr = RapidOCR()
            with Image.open(file_path) as pil_img:
                img_arr = np.array(pil_img.convert("RGB"))
            out = ocr(img_arr)
            if out and out.txts:
                return "\n".join(out.txts)
        except Exception as exc:
            print(f"[Image Extraction] RapidOCR warning: {exc}")
        return ""

    # Handle PDF formats
    pdf = None
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(file_path)
        text_pages = [page.get_textpage().get_text_range().strip() for page in pdf]
        digital_text = "\n\n".join([t for t in text_pages if t])
        if len(digital_text) > 150:
            return digital_text

        # Scanned PDF: Perform OCR on rendered pages
        from rapidocr import RapidOCR
        ocr = RapidOCR()
        ocr_lines = []
        for page in pdf:
            img = page.render(scale=2).to_numpy()
            out = ocr(img)
            if out and out.txts:
                ocr_lines.extend(out.txts)
        ocr_text = "\n".join(ocr_lines)
        if ocr_text.strip():
            return ocr_text
    except Exception as exc:
        print(f"[PDF Extraction] Direct/OCR extraction warning: {exc}")
    finally:
        if pdf is not None:
            try:
                pdf.close()
            except Exception:
                pass

    try:
        converter = DocumentConverter()
        return converter.convert(file_path).document.export_to_markdown()
    except Exception as exc:
        print(f"[PDF Extraction] Docling fallback failed: {exc}")
        return ""
