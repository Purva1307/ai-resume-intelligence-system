import streamlit as st
import matplotlib.pyplot as plt

from src.extractor import extract_text
from src.preprocess import preprocess_text
from src.matcher import extract_skills
from src.scorer import (
    compute_structured_score,
    compute_hybrid_score,
    get_missing_skills_with_severity,
    compute_category_scores,
)
from src.semantic import semantic_skill_match, load_model
from src.visualiser import generate_similarity_matrix, plot_heatmap


# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide",
)

# ==============================
# Header
# ==============================
st.markdown(
    """
    <h1 style='text-align: center;'>AI Resume vs Job Description Matcher</h1>
    <p style='text-align: center; color: gray;'>
    Instantly analyze resume-job fit using AI skill intelligence
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Cached Semantic Model
# ==============================
@st.cache_resource(show_spinner=False)
def get_semantic_model():
    return load_model()


# ==============================
# Inputs
# ==============================
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_text = st.text_area("Job Description", height=150)

use_semantic = st.toggle("Enable Semantic Matching", value=True)

st.markdown("<br>", unsafe_allow_html=True)

center_col = st.columns([1, 2, 1])[1]
analyze_clicked = center_col.button("Analyze Resume", use_container_width=True)


# ==============================
# Main Logic
# ==============================
if analyze_clicked:

    if uploaded_file is not None and jd_text.strip():

        with st.spinner("Analyzing resume..."):

            # Save resume
            with open("temp_resume.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # Resume processing
            raw_text = extract_text("temp_resume.pdf")
            clean_text = preprocess_text(raw_text)
            resume_skills = extract_skills(clean_text)

            # JD processing
            jd_clean = preprocess_text(jd_text)
            jd_skills = extract_skills(jd_clean)

            # Structured Score
            structured_score, matched_structured = compute_structured_score(
                resume_skills,
                jd_skills,
            )

            # Semantic Score
            semantic_score = 0.0
            semantic_matched = set()
            model = None

            if use_semantic:
                model = get_semantic_model()
                semantic_matched, semantic_score = semantic_skill_match(
                    model,
                    resume_skills,
                    jd_skills,
                )

            # Hybrid Fusion
            final_score = compute_hybrid_score(
                structured_score,
                semantic_score,
            )

            matched = sorted(
                set(matched_structured).union(semantic_matched)
            )

            severity = get_missing_skills_with_severity(
                resume_skills,
                jd_skills,
            )

            category_scores = compute_category_scores(
                resume_skills,
                jd_skills,
            )

        # ================= KPI =================
        k1, k2, k3 = st.columns(3)

        with k1:
            st.metric("Final Match Score", f"{final_score:.2f}%")

        with k2:
            st.metric("Exact Skill Match", f"{structured_score:.2f}%")

        with k3:
            st.metric("Related Skill Match", f"{semantic_score:.2f}%")

        st.progress(int(max(0, min(final_score, 100))))

        # ================= MATCH BREAKDOWN =================
        st.markdown("### Match Breakdown For This Job")

        st.markdown(
            f"""
**Exact Skill Match:** {structured_score:.2f}%  

**Related Skill Match:** {semantic_score:.2f}%  

**Overall Fit Score:** {final_score:.2f}%  
"""
        )

        if final_score >= 75:
            st.success("The resume is a strong fit for this role.")
        elif final_score >= 50:
            st.warning("The resume matches many requirements, but some gaps remain.")
        else:
            st.error("The resume needs more alignment with this job description.")

        # ================= DASHBOARD =================
        st.subheader("Skill Analysis Dashboard")

        dashboard_col1, dashboard_col2 = st.columns(2)

        # ---------- PIE ----------
        with dashboard_col1:

            labels = []
            sizes = []

            for cat, val in category_scores.items():
                if val > 0:
                    labels.append(cat.upper())
                    sizes.append(val)

            if sizes:
                fig, ax = plt.subplots(figsize=(3, 3))
                fig.patch.set_facecolor("#0E1117")
                ax.set_facecolor("#0E1117")

                pie_output = ax.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.0f%%",
                    startangle=90,
                    wedgeprops=dict(linewidth=1, edgecolor="#111"),
                )

                if len(pie_output) == 3:
                    wedges, texts, autotexts = pie_output
                else:
                    wedges, texts = pie_output
                    autotexts = []

                for text in texts:
                    text.set_color("white")
                    text.set_fontsize(8)

                for autotext in autotexts:
                    autotext.set_color("white")
                    autotext.set_fontsize(8)
                    autotext.set_fontweight("bold")

                ax.axis("equal")
                st.pyplot(fig, clear_figure=True)

        # ---------- HEATMAP ----------
        with dashboard_col2:

            if use_semantic and model is not None:

                resume_flat = [
                    skill for skills in resume_skills.values() for skill in skills
                ]

                jd_flat = [
                    skill for skills in jd_skills.values() for skill in skills
                ]

                if resume_flat and jd_flat:
                    similarity_matrix = generate_similarity_matrix(
                        model,
                        resume_flat,
                        jd_flat,
                    )

                    if similarity_matrix is not None:
                        fig = plot_heatmap(
                            similarity_matrix,
                            resume_flat,
                            jd_flat,
                        )
                        st.pyplot(fig)

        # ================= RESULTS =================
        st.markdown("<br>", unsafe_allow_html=True)

        results_col1, results_col2 = st.columns(2)

        with results_col1:
            st.subheader("Matched Skills")

            if matched:
                for skill in matched:
                    st.markdown(f"- {skill.title()}")
            else:
                st.info("No matched skills found")

        with results_col2:
            st.subheader("Skill Gap Analysis")

            if severity.get("critical"):
                st.error("Critical skills missing")
                for s in severity["critical"]:
                    st.markdown(f"- {s.title()}")

            if severity.get("medium"):
                st.warning("Medium priority skills missing")
                for s in severity["medium"]:
                    st.markdown(f"- {s.title()}")

            if severity.get("low"):
                st.info("Low priority skills missing")
                for s in severity["low"]:
                    st.markdown(f"- {s.title()}")

            if (
                not severity.get("critical")
                and not severity.get("medium")
                and not severity.get("low")
            ):
                st.success("No major skill gaps detected")

        # ================= EXPANDER =================
        with st.expander("View Detected Skills"):

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Resume Skills**")
                if resume_skills:
                    for cat, skills in resume_skills.items():
                        st.markdown(f"**{cat.upper()}**")
                        for s in skills:
                            st.markdown(f"- {s.title()}")

            with c2:
                st.markdown("**JD Skills**")
                if jd_skills:
                    for cat, skills in jd_skills.items():
                        st.markdown(f"**{cat.upper()}**")
                        for s in skills:
                            st.markdown(f"- {s.title()}")

    else:
        st.warning("Please upload resume and paste job description.")