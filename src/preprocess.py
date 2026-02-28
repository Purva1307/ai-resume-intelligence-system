import re
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    """Clean and normalize resume text."""

    # lower case
    text = text.lower()

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # remove special characters but keep + and #
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)

    return text.strip()