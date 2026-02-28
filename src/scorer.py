# src/scorer.py

from typing import Dict, List, Tuple

# ==============================
# Category importance weights
# ==============================
WEIGHTS = {
    "programming": 0.35,
    "ai_ml": 0.30,
    "data": 0.20,
    "tools": 0.10,
    "web": 0.05,
}

# Hybrid fusion weights
STRUCTURED_WEIGHT = 0.6
SEMANTIC_WEIGHT = 0.4


# ==============================
# Structured deterministic scoring
# ==============================
def compute_structured_score(
    resume_skills: Dict[str, List[str]],
    jd_skills: Dict[str, List[str]],
) -> Tuple[float, List[str]]:

    total_weight = 0.0
    earned_weight = 0.0
    matched_skills = []

    for category, jd_list in jd_skills.items():

        weight = WEIGHTS.get(category, 0.05)
        total_weight += weight

        resume_list = resume_skills.get(category, [])

        if not jd_list:
            continue

        common = set(resume_list) & set(jd_list)
        category_score = len(common) / len(jd_list)

        earned_weight += weight * category_score
        matched_skills.extend(common)

    if total_weight == 0:
        return 0.0, []

    structured_score = (earned_weight / total_weight) * 100

    return round(structured_score, 2), sorted(set(matched_skills))


# ==============================
# Hybrid fusion scoring
# ==============================
def compute_hybrid_score(
    structured_score: float,
    semantic_score: float,
) -> float:

    final_score = (
        STRUCTURED_WEIGHT * structured_score
        + SEMANTIC_WEIGHT * semantic_score
    )

    return round(final_score, 2)


# ==============================
# Missing skills with severity
# ==============================
def get_missing_skills_with_severity(
    resume_skills: Dict[str, List[str]],
    jd_skills: Dict[str, List[str]],
):

    critical, medium, low = [], [], []

    for category, jd_list in jd_skills.items():

        resume_list = resume_skills.get(category, [])
        diff = set(jd_list) - set(resume_list)

        weight = WEIGHTS.get(category, 0.05)

        for skill in diff:
            if weight >= 0.30:
                critical.append(skill)
            elif weight >= 0.15:
                medium.append(skill)
            else:
                low.append(skill)

    return {
        "critical": sorted(set(critical)),
        "medium": sorted(set(medium)),
        "low": sorted(set(low)),
    }


# ==============================
# Category coverage (for pie)
# ==============================
def compute_category_scores(
    resume_skills: Dict[str, List[str]],
    jd_skills: Dict[str, List[str]],
):

    category_scores = {}

    for category, jd_list in jd_skills.items():

        if not jd_list:
            continue

        resume_list = resume_skills.get(category, [])
        common = set(resume_list) & set(jd_list)

        coverage = (len(common) / len(jd_list)) * 100
        category_scores[category] = round(coverage, 2)

    return category_scores