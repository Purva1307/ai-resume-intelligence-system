# src/semantic.py

from typing import Dict, List, Set, Tuple
import torch
from sentence_transformers import SentenceTransformer, util

SIM_THRESHOLD = 0.65


def load_model():
    """
    Lazy load model.
    Called from Streamlit with caching.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_skill_match(
    model,
    resume_skills: Dict[str, List[str]],
    jd_skills: Dict[str, List[str]],
) -> Tuple[Set[str], float]:

    resume_flat = [
        skill for skills in resume_skills.values() for skill in skills
    ]
    jd_flat = [
        skill for skills in jd_skills.values() for skill in skills
    ]

    if not resume_flat or not jd_flat:
        return set(), 0.0

    resume_embeddings = model.encode(resume_flat, convert_to_tensor=True)
    jd_embeddings = model.encode(jd_flat, convert_to_tensor=True)

    if not isinstance(resume_embeddings, torch.Tensor):
        resume_embeddings = torch.tensor(resume_embeddings)

    if not isinstance(jd_embeddings, torch.Tensor):
        jd_embeddings = torch.tensor(jd_embeddings)

    if resume_embeddings.dim() == 1:
        resume_embeddings = resume_embeddings.unsqueeze(0)

    if jd_embeddings.dim() == 1:
        jd_embeddings = jd_embeddings.unsqueeze(0)

    similarity_matrix = util.cos_sim(jd_embeddings, resume_embeddings)

    matched = set()

    for i in range(similarity_matrix.shape[0]):
        max_similarity = float(torch.max(similarity_matrix[i]).item())
        if max_similarity >= SIM_THRESHOLD:
            matched.add(jd_flat[i])

    semantic_score = (len(matched) / len(jd_flat)) * 100

    return matched, round(semantic_score, 2)