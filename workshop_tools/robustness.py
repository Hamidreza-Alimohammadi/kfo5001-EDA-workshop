# Copyright (c) 2026 Hamidreza Alimohammadi and contributing rights holders.
# All rights reserved. See COPYRIGHT.md.

import numpy as np
import pandas as pd

from workshop_tools import trajectories
from workshop_tools.patterns import (
    build_pattern_matrix,
    standardize_pattern_matrix,
)


def build_pattern_views(epoch_features, views, epoch="Late-Tone"):
    """Build alternative subject representations on their shared subjects."""
    if not views:
        raise ValueError("Choose at least one pattern view.")

    matrices = {
        label: build_pattern_matrix(
            epoch_features,
            signals=signals,
            epoch=epoch,
        )
        for label, signals in views.items()
    }
    complete_subjects = [
        set(matrix.dropna().index)
        for matrix in matrices.values()
    ]
    common_subjects = sorted(set.intersection(*complete_subjects))
    if len(common_subjects) < 2:
        raise ValueError("Fewer than two subjects are complete across all views.")

    standardized = {
        label: standardize_pattern_matrix(matrix.loc[common_subjects])
        for label, matrix in matrices.items()
    }
    return standardized, common_subjects


def _rms_distances(standardized_matrix):
    values = standardized_matrix.to_numpy(dtype=float)
    differences = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.mean(differences**2, axis=2))


def plot_view_distances(standardized_views, figsize=(11, 8)):
    """Compare subject distances under alternative feature representations."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage

    if not standardized_views:
        raise ValueError("Choose at least one pattern view.")

    labels = list(standardized_views)
    reference = standardized_views[labels[0]]
    subject_order = leaves_list(linkage(reference, method="ward"))
    subjects = reference.index[subject_order]
    distances = {
        label: pd.DataFrame(
            _rms_distances(matrix),
            index=matrix.index,
            columns=matrix.index,
        ).loc[subjects, subjects]
        for label, matrix in standardized_views.items()
    }
    vmax = max(distance.to_numpy().max() for distance in distances.values())

    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for ax, label in zip(axes.flat, labels):
        image = ax.imshow(
            distances[label],
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        short_subjects = [subject[-2:] for subject in subjects]
        ax.set_xticks(np.arange(len(subjects)))
        ax.set_xticklabels(short_subjects, fontsize=7)
        ax.set_yticks(np.arange(len(subjects)))
        ax.set_yticklabels(short_subjects, fontsize=7)
        ax.set_title(label)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Subject")

    for ax in axes.flat[len(labels) :]:
        ax.set_visible(False)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("Standardized root-mean-square distance")
    fig.suptitle("Do subject relationships survive feature choices?")
    return fig, axes


def trial_omission_coclustering(
    epoch_features,
    signals=("freezing", "hr"),
    epoch="Late-Tone",
    n_clusters=2,
):
    """Measure co-clustering after omitting each trial position in turn."""
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    from scipy.cluster.hierarchy import fcluster, linkage

    baseline = build_pattern_matrix(
        epoch_features,
        signals=signals,
        epoch=epoch,
    ).dropna()
    subjects = baseline.index
    together = np.zeros((len(subjects), len(subjects)), dtype=float)
    analysis_count = 0

    for phase in trajectories.PHASE_ORDER:
        phase_trials = sorted(
            epoch_features.loc[epoch_features["phase"] == phase, "trial"].unique()
        )
        for trial in phase_trials:
            omit_trial = (
                (epoch_features["phase"] == phase)
                & (epoch_features["trial"] == trial)
            )
            reduced_features = epoch_features.loc[~omit_trial]
            matrix = build_pattern_matrix(
                reduced_features,
                signals=signals,
                epoch=epoch,
            ).reindex(subjects)
            if matrix.isna().any().any():
                continue

            standardized = standardize_pattern_matrix(matrix)
            labels = fcluster(
                linkage(standardized, method="ward"),
                t=n_clusters,
                criterion="maxclust",
            )
            together += labels[:, None] == labels[None, :]
            analysis_count += 1

    if analysis_count == 0:
        raise ValueError("No complete leave-one-trial-out analyses were available.")
    return pd.DataFrame(
        together / analysis_count,
        index=subjects,
        columns=subjects,
    ), analysis_count


def plot_coclustering_stability(coclustering, analysis_count, figsize=(7, 6)):
    """Plot how consistently each subject pair shares a candidate branch."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dissimilarity = 1 - coclustering
    order = leaves_list(
        linkage(squareform(dissimilarity, checks=False), method="average")
    )
    ordered = coclustering.iloc[order, order]
    labels = [subject[-2:] for subject in ordered.index]

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(ordered, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Subject")
    ax.set_title(
        f"Two-branch co-membership across {analysis_count} trial omissions"
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Fraction assigned to the same branch")
    fig.tight_layout()
    return fig, ax
