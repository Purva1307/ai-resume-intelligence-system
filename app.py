import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

from src.extractor import extract_text
from src.preprocess import preprocess_text
from src.matcher import extract_skills
from src.scorer import compute_match_score


if __name__ == "__main__":
    file_path = "test_data/resume.pdf"

    # ==============================
    # Resume processing
    # ==============================
    raw_text = extract_text(file_path)
    clean_text = preprocess_text(raw_text)
    resume_skills = extract_skills(clean_text)

    print("\n EXTRACTED RESUME SKILLS:")
    print(resume_skills)

    # ==============================
    # ⭐ ADD YOUR JD TEXT HERE ⭐
    # ==============================
    jd_text = """
Looking for ML engineer with JS and CSS3 experience.
"""

    jd_clean = preprocess_text(jd_text)
    jd_skills = extract_skills(jd_clean)

    print("\n JD SKILLS:")
    print(jd_skills)

    # ==============================
    # Match score
    # ==============================
    score, matched_skills = compute_match_score(resume_skills, jd_skills)

def get_missing_skills(resume_skills: dict, jd_skills: dict):
    missing = []

    for category, jd_list in jd_skills.items():
        resume_list = resume_skills.get(category, [])
        diff = set(jd_list) - set(resume_list)
        missing.extend(list(diff))

    return sorted(missing) 

    print("\n MATCH SCORE:")
    print(score, "%")

    print("\n MATCHED SKILLS:")
    print(matched_skills)