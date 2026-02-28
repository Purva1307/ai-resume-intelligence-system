# src/matcher.py

import re
from difflib import get_close_matches

# ==============================
# Master skill database
# ==============================
SKILL_DB = {
    "programming": [
        "python", "java", "c++", "c", "javascript", "typescript"
    ],
    "ai_ml": [
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "nlp",
        "computer vision",
        "scikit-learn",
        "large language model",
        "llm",
    ],
    "data": [
        "pandas", "numpy", "matplotlib", "seaborn",
        "data analysis", "data science"
    ],
    "tools": [
        "git", "github", "docker", "linux", "vscode"
    ],
    "web": [
        "html", "css", "react", "node", "express"
    ],
    "emerging": [
        "blockchain", "web3"
    ]
}

# ==============================
# Synonyms (semantic awareness)
# ==============================
SYNONYMS = {
    "machine learning": ["ml"],
    "large language model": ["llm", "large language models"],
    "javascript": ["js"],
    "css": ["css3"],
    "html": ["html5"],
}

# ==============================
# Normalizer
# ==============================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9+#.\s]', ' ', text)
    return text


# ==============================
# Fuzzy helper
# ==============================
def fuzzy_contains(text_words, skill):
    matches = get_close_matches(skill, text_words, n=1, cutoff=0.88)
    return len(matches) > 0


# ==============================
# Skill extractor (SAFE + FUZZY)
# ==============================
def extract_skills(text: str) -> dict:
    text = normalize_text(text)
    words = set(text.split())

    found_skills = {}

    for category, skills in SKILL_DB.items():
        matched = []

        for skill in skills:
            skill_matched = False

            # ✅ strict word boundary (prevents java/javascript bug)
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text):
                skill_matched = True

            # ✅ synonym check
            if not skill_matched and skill in SYNONYMS:
                for alias in SYNONYMS[skill]:
                    alias_pattern = r"\b" + re.escape(alias) + r"\b"
                    if re.search(alias_pattern, text):
                        skill_matched = True
                        break

            # ✅ fuzzy fallback (Phase-3 intelligence)
            if not skill_matched:
                if fuzzy_contains(words, skill):
                    skill_matched = True

            if skill_matched:
                matched.append(skill)

        if matched:
            found_skills[category] = sorted(set(matched))

    return found_skills