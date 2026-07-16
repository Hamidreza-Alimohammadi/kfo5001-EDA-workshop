# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

import numpy as np
import pandas as pd

from workshop_tools.trajectories import summarize_trajectories


SIGNAL_FEATURES = {
    "freezing": "freezing_fraction",
    "longest_bout": "longest_freezing_bout_s",
    "motion": "motion_median",
    "speed": "speed_median_px_s",
    "hr": "hr_median_bpm",
}
TRIAL_PHASES = ["ext_1", "ext_2"]
STATE_FEATURES = (
    "freezing_fraction",
    "longest_freezing_bout_s",
    "motion_median",
    "speed_median_px_s",
)


def build_pattern_matrix(
    epoch_features,
    signals=("freezing", "motion"),
    epoch="Late-Tone",
):
    """Build one subject row from four temporal traits per selected signal."""
    unknown = set(signals) - set(SIGNAL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown signals: {sorted(unknown)}")
    if not signals:
        raise ValueError("Choose at least one signal.")

    matrices = []
    for signal in signals:
        summary = summarize_trajectories(
            epoch_features,
            feature=SIGNAL_FEATURES[signal],
            epoch=epoch,
        )
        day_1 = summary.loc[summary["phase"] == "ext_1"].set_index(
            "subject_id"
        )
        day_2 = summary.loc[summary["phase"] == "ext_2"].set_index(
            "subject_id"
        )
        matrix = pd.DataFrame(
            {
                f"{signal}_d1_early": day_1["early_block_median"],
                f"{signal}_d1_trend": day_1["linear_trend_per_trial"],
                f"{signal}_d2_trend": day_2["linear_trend_per_trial"],
                f"{signal}_between_days": day_1[
                    "day2_early_minus_day1_late"
                ],
            }
        )
        matrices.append(matrix)

    return pd.concat(matrices, axis=1).sort_index()


def build_trial_matrix(epoch_features, signals=("freezing", "motion"), epoch="Late-Tone"):
    """Build one subject row from every trial value at a selected epoch."""
    unknown = set(signals) - set(SIGNAL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown signals: {sorted(unknown)}")
    if not signals:
        raise ValueError("Choose at least one signal.")
    selected = epoch_features.loc[epoch_features["epoch"] == epoch]
    if selected.empty:
        raise ValueError(f"No rows are available for epoch: {epoch}")
    feature_columns = [SIGNAL_FEATURES[signal] for signal in signals]
    matrix = selected.pivot(index="subject_id", columns=["phase", "trial"], values=feature_columns)
    ordered_columns = pd.MultiIndex.from_tuples(
        [
            (SIGNAL_FEATURES[signal], phase, trial)
            for signal in signals
            for phase in TRIAL_PHASES
            for trial in range(1, 17)
        ],
        names=matrix.columns.names,
    )
    matrix = matrix.reindex(columns=ordered_columns)
    signal_names = {feature: signal for signal, feature in SIGNAL_FEATURES.items()}
    matrix.columns = [
        f"{signal_names[feature]}_{'d1' if phase == 'ext_1' else 'd2'}_t{trial:02d}"
        for feature, phase, trial in matrix.columns
    ]
    return matrix.sort_index()


def build_state_matrix(epoch_features, feature_columns=None):
    """Build pooled epoch observations while retaining repeated-measure labels."""
    if feature_columns is None:
        feature_columns = STATE_FEATURES
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("Choose at least one state feature.")
    missing_columns = sorted(set(feature_columns) - set(epoch_features.columns))
    if missing_columns:
        raise ValueError(f"Unknown state features: {missing_columns}")
    metadata_columns = ["subject_id", "phase", "trial", "epoch"]
    state_matrix = epoch_features.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    state_metadata = epoch_features.loc[:, metadata_columns].copy()
    return state_matrix, state_metadata


def standardize_pattern_matrix(pattern_matrix):
    """Z-standardize subject features while preserving row and column labels."""
    from sklearn.preprocessing import StandardScaler

    if pattern_matrix.isna().any().any():
        raise ValueError("Resolve missing values before standardization.")
    scaler = StandardScaler()
    values = scaler.fit_transform(pattern_matrix)
    return pd.DataFrame(values, index=pattern_matrix.index, columns=pattern_matrix.columns)


def fit_pattern_pca(standardized_matrix):
    """Fit two PCA dimensions and return labeled scores, loadings, and variance."""
    from sklearn.decomposition import PCA

    if len(standardized_matrix) < 2 or standardized_matrix.shape[1] < 2:
        raise ValueError("PCA requires at least two subjects and two features.")
    pca = PCA(n_components=2)
    score_values = pca.fit_transform(standardized_matrix)
    scores = pd.DataFrame(
        score_values,
        index=standardized_matrix.index,
        columns=["PC1", "PC2"],
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=standardized_matrix.columns,
        columns=["PC1", "PC2"],
    )
    return scores, loadings, pca.explained_variance_ratio_


def fit_state_umap(standardized_matrix, n_neighbors=30, min_dist=0.20, random_state=42):
    """Fit a reproducible two-dimensional UMAP and assess local fidelity."""
    import umap
    from sklearn.manifold import trustworthiness

    if standardized_matrix.isna().any().any():
        raise ValueError("Resolve missing state features before fitting UMAP.")
    if len(standardized_matrix) < 6 or standardized_matrix.shape[1] < 2:
        raise ValueError("State UMAP requires six observations and two features.")
    if not 2 <= n_neighbors < len(standardized_matrix):
        raise ValueError("n_neighbors must be at least 2 and smaller than the observation count.")
    if not 0 <= min_dist <= 1:
        raise ValueError("min_dist must lie between 0 and 1.")

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean", random_state=random_state, n_jobs=1)
    coordinates = reducer.fit_transform(standardized_matrix)
    embedding = pd.DataFrame(coordinates, index=standardized_matrix.index, columns=["UMAP1", "UMAP2"])
    diagnostic_neighbors = min(10, (len(standardized_matrix) - 1) // 2)
    diagnostics = pd.Series({
        "observations": len(standardized_matrix),
        "features": standardized_matrix.shape[1],
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "trustworthiness": trustworthiness(standardized_matrix, embedding, n_neighbors=diagnostic_neighbors),
    })
    return embedding, diagnostics


def plot_feature_correlation(pattern_matrix, figsize=(7, 6)):
    """Show correlation and redundancy among subject-level temporal traits."""
    import matplotlib.pyplot as plt

    correlation = pattern_matrix.corr()
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = correlation.columns.str.replace("_", " ")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlation.iloc[row, column]
            color = "white" if abs(value) > 0.55 else "0.15"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Pearson correlation")
    ax.set_title(f"Temporal-trait correlation (n = {len(pattern_matrix)} subjects)")
    fig.tight_layout()
    return fig, ax


def top_feature_correlations(pattern_matrix, top_n=10):
    """List the strongest unique Pearson correlations between feature pairs."""
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    correlation = pattern_matrix.corr()
    upper_triangle = np.triu(np.ones(correlation.shape, dtype=bool), k=1)
    pairs = correlation.where(upper_triangle).stack().rename("pearson_r").reset_index()
    pairs.columns = ["feature_1", "feature_2", "pearson_r"]
    pairs["absolute_r"] = pairs["pearson_r"].abs()
    return pairs.sort_values("absolute_r", ascending=False).head(top_n).reset_index(drop=True)


def plot_pattern_pca(scores, loadings, explained_variance, figsize=(11, 4.8)):
    """Show subject PCA scores beside the feature loadings that define the axes."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].scatter(
        scores["PC1"],
        scores["PC2"],
        s=115,
        color="#326b8c",
        edgecolor="white",
        linewidth=0.8,
    )
    for subject_id, row in scores.iterrows():
        axes[0].text(
            row["PC1"],
            row["PC2"],
            subject_id.replace("subject_", "", 1),
            ha="center",
            va="center",
            color="white",
            fontsize=6.5,
            fontweight="bold",
        )
    axes[0].axhline(0, color="0.8", linewidth=0.8)
    axes[0].axvline(0, color="0.8", linewidth=0.8)
    axes[0].set_xlabel(f"PC1 ({explained_variance[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({explained_variance[1]:.1%})")
    axes[0].set_title("Subjects in PCA space")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    limit = np.abs(loadings.to_numpy()).max()
    image = axes[1].imshow(loadings, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["PC1", "PC2"])
    axes[1].set_yticks(np.arange(len(loadings)))
    axes[1].set_yticklabels(loadings.index.str.replace("_", " "), fontsize=8)
    for row in range(len(loadings)):
        for column in range(2):
            axes[1].text(
                column,
                row,
                f"{loadings.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=axes[1], shrink=0.82)
    colorbar.set_label("Loading")
    axes[1].set_title("Features defining each component")
    fig.tight_layout()
    return fig, axes


def plot_hierarchical_patterns(standardized_matrix, figsize=(11, 6.5)):
    """Show a Ward dendrogram aligned with the standardized subject matrix."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    if len(standardized_matrix) < 2:
        raise ValueError("Hierarchical clustering requires at least two subjects.")
    linkage_matrix = linkage(standardized_matrix.to_numpy(), method="ward")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1, 2.6]},
        constrained_layout=True,
    )
    tree = dendrogram(
        linkage_matrix,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="0.25",
        ax=axes[0],
    )
    axes[0].set_ylabel("Ward distance")
    axes[0].set_title("Hierarchical similarity between subjects")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    subject_order = standardized_matrix.index[tree["leaves"]].tolist()
    ordered = standardized_matrix.loc[subject_order]
    image = axes[1].imshow(
        ordered.T,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=2.5,
        extent=[0, 10 * len(subject_order), len(ordered.columns) - 0.5, -0.5],
    )
    axes[1].set_xticks(5 + 10 * np.arange(len(subject_order)))
    axes[1].set_xticklabels(subject_order, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(np.arange(len(ordered.columns)))
    axes[1].set_yticklabels(ordered.columns.str.replace("_", " "), fontsize=8)
    axes[1].set_xlabel("Subject")
    axes[1].set_ylabel("Standardized temporal trait")
    colorbar = fig.colorbar(image, ax=axes[1], shrink=0.82, pad=0.02)
    colorbar.set_label("Z-score")
    return fig, axes, subject_order


def plot_trial_patterns(standardized_matrix, figsize=(13, 6)):
    """Show raw-trial fingerprints and PCA from the same standardized matrix."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage

    if standardized_matrix.isna().any().any():
        raise ValueError("Resolve missing trial values before plotting.")
    if len(standardized_matrix) < 3 or standardized_matrix.shape[1] < 2:
        raise ValueError("Trial-pattern analysis requires three subjects and two features.")
    if standardized_matrix.shape[1] % 16:
        raise ValueError("Trial-pattern columns must contain complete 16-trial blocks.")

    subject_order = leaves_list(linkage(standardized_matrix, method="ward"))
    ordered = standardized_matrix.iloc[subject_order]
    scores, _, explained = fit_pattern_pca(standardized_matrix)
    block_starts = list(range(0, standardized_matrix.shape[1], 16))
    block_labels = [standardized_matrix.columns[start].rsplit("_t", 1)[0].replace("_", " ") for start in block_starts]

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.4, 1]})
    image = axes[0].imshow(ordered, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    for boundary in block_starts[1:]:
        axes[0].axvline(boundary - 0.5, color="white", linewidth=1.2)
    axes[0].set_xticks([start + 7.5 for start in block_starts])
    axes[0].set_xticklabels(block_labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_yticks(np.arange(len(ordered)))
    axes[0].set_yticklabels(ordered.index, fontsize=8)
    axes[0].set_xlabel("Trial blocks at the selected epoch")
    axes[0].set_ylabel("Subject (Ward ordered)")
    axes[0].set_title("Standardized 32-trial fingerprints")
    colorbar = fig.colorbar(image, ax=axes[0], shrink=0.82, pad=0.02)
    colorbar.set_label("Trial-specific z-score")

    axes[1].scatter(scores["PC1"], scores["PC2"], s=115, color="#326b8c", edgecolor="white", linewidth=0.8)
    for subject_id, row in scores.iterrows():
        axes[1].text(row["PC1"], row["PC2"], subject_id[-2:], ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
    axes[1].axhline(0, color="0.8", linewidth=0.8)
    axes[1].axvline(0, color="0.8", linewidth=0.8)
    axes[1].set_xlabel(f"PC1 ({explained[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({explained[1]:.1%})")
    axes[1].set_title("PCA of the same trial features")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.suptitle(f"Exploratory high-dimensional lens: {len(standardized_matrix)} subjects x {standardized_matrix.shape[1]} trial features")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, axes, ordered.index.tolist()


def plot_state_umap(embedding, metadata, figsize=(12, 9)):
    """Colour one pooled state embedding by its repeated-measure labels."""
    import matplotlib.pyplot as plt
    from workshop_tools import style

    if not embedding.index.equals(metadata.index):
        raise ValueError("Embedding and metadata rows must be aligned.")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    epoch_colors = dict(zip(style.EPOCH_LABELS, style.EPOCH_COLORS))
    for epoch in style.EPOCH_LABELS:
        selected = metadata["epoch"].astype(str) == epoch
        axes[0, 0].scatter(embedding.loc[selected, "UMAP1"], embedding.loc[selected, "UMAP2"], s=16, alpha=0.65, color=epoch_colors[epoch], label=epoch, edgecolors="none")
    axes[0, 0].legend(frameon=False, fontsize=8, markerscale=1.4)
    axes[0, 0].set_title("Colour by epoch")

    for phase in TRIAL_PHASES:
        selected = metadata["phase"] == phase
        axes[0, 1].scatter(embedding.loc[selected, "UMAP1"], embedding.loc[selected, "UMAP2"], s=16, alpha=0.65, color=style.PHASE_COLORS[phase], label=style.PHASE_LABELS[phase], edgecolors="none")
    axes[0, 1].legend(frameon=False, fontsize=8, markerscale=1.4)
    axes[0, 1].set_title("Colour by extinction day")

    trial_points = axes[1, 0].scatter(embedding["UMAP1"], embedding["UMAP2"], c=metadata["trial"], s=16, alpha=0.7, cmap="viridis", edgecolors="none")
    fig.colorbar(trial_points, ax=axes[1, 0], label="Trial within day")
    axes[1, 0].set_title("Colour by trial progression")

    subjects = sorted(metadata["subject_id"].unique())
    subject_colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(subjects)))
    for subject, color in zip(subjects, subject_colors):
        selected = metadata["subject_id"] == subject
        axes[1, 1].scatter(embedding.loc[selected, "UMAP1"], embedding.loc[selected, "UMAP2"], s=14, alpha=0.6, color=color, label=subject.replace("subject_", ""), edgecolors="none")
    axes[1, 1].legend(title="Subject", frameon=False, fontsize=7, title_fontsize=8, ncol=2, markerscale=1.4)
    axes[1, 1].set_title("Colour by subject")

    for ax in axes.flat:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        style.despine(ax)
    fig.suptitle("Pooled extinction-state geometry: repeated views of one embedding")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes
