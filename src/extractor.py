import pdfplumber
from docx import Document
from pathlib import Path


def extract_text(file_path: str) -> str:
    """
    Extract raw text from PDF or DOCX resume.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        return _extract_from_pdf(path)

    elif path.suffix.lower() == ".docx":
        return _extract_from_docx(path)

    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")


def _extract_from_pdf(path: Path) -> str:
    text = ""

    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text.strip()


def _extract_from_docx(path: Path) -> str:
    doc = Document(str(path))
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return text.strip()