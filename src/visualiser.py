# src/visualiser.py

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import util


def generate_similarity_matrix(model, resume_flat, jd_flat):
    """
    Generates cosine similarity matrix between JD skills (rows)
    and Resume skills (columns).
    """

    if not resume_flat or not jd_flat:
        return None

    try:
        resume_embeddings = model.encode(
            resume_flat,
            convert_to_tensor=True,
        )

        jd_embeddings = model.encode(
            jd_flat,
            convert_to_tensor=True,
        )

        similarity_tensor = util.cos_sim(
            jd_embeddings,
            resume_embeddings,
        )

        return similarity_tensor.detach().cpu().numpy()

    except Exception as e:
        print("Heatmap error:", e)
        return None


def plot_heatmap(similarity_matrix, resume_flat, jd_flat):
    """
    Dark-themed heatmap
    """

    if similarity_matrix is None:
        return None

    # Dark figure
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    heatmap = ax.imshow(
        similarity_matrix,
        aspect="auto",
        cmap="viridis",
    )

    ax.set_xticks(range(len(resume_flat)))
    ax.set_yticks(range(len(jd_flat)))

    ax.set_xticklabels(
        resume_flat,
        rotation=45,
        ha="right",
        fontsize=8,
        color="white",
    )

    ax.set_yticklabels(
        jd_flat,
        fontsize=8,
        color="white",
    )

    ax.set_xlabel("Resume Skills", color="white")
    ax.set_ylabel("Job Description Skills", color="white")
    ax.set_title("Semantic Similarity Heatmap", color="white")

    # White ticks
    ax.tick_params(colors="white")

    # Dark colorbar
    cbar = fig.colorbar(heatmap)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    fig.tight_layout()

    return fig