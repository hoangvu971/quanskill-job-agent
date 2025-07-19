import pdfplumber
import re
from pathlib import Path


class ResumeParser:
    """
    A robust PDF resume parser that extracts plain text.
    """

    def __init__(self, resume_path: str):
        self.resume_path = Path(resume_path)
        if not self.resume_path.exists():
            raise FileNotFoundError(f"Resume file not found: {resume_path}")

    def extract_text(self) -> str:
        """
        Extract text from all pages of the resume PDF.
        Returns cleaned text.
        """
        text_chunks = []
        with pdfplumber.open(self.resume_path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text_chunks.append(content)

        full_text = "\n".join(text_chunks)
        return self._clean_text(full_text)

    def _clean_text(self, text: str) -> str:
        """
        Basic cleanup: remove excessive spaces, blank lines, etc.
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
