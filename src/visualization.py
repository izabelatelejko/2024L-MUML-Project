"""Module for visualization utils."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import to_rgb


def _enlighten(color):
    r, g, b = to_rgb(color)
    return (r, g, b, 0.05)


def plot_results(df):
    """Plot boxplots showing index of succes for different feature selection methods."""
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.boxplot(
        data=df,
        x="names",
        y="scores",
        fill=True,
        gap=0.1,
        color=(0.125, 0.125, 0.875),
        ax=ax,
    )

    num_artists = len(ax.patches)
    num_lines = len(ax.lines)
    lines_per_artist = num_lines // num_artists

    for i, artist in enumerate(ax.patches):
        color = artist.get_facecolor()
        lcolor = _enlighten(color)
        artist.set_color(lcolor)
        artist.set_edgecolor(color)
        for j in range(lines_per_artist):
            ax.lines[i * lines_per_artist + j].set_color(color)

    plt.title("")
    plt.show()


def plot_all_datasets(accuracy_scores, model_name):
    """Plot boxplots showing accuracy for different feature selection methods on multiple datasets."""
    datasets = [key for key, _ in accuracy_scores.items()]

    fig, ax = plt.subplots(3, 3, figsize=(14, 14), sharey=True)
    for i, dataset_name in enumerate(datasets):
        names = []
        scores = []
        for method, acc in accuracy_scores[dataset_name].items():
            names = names + [method for _ in range(acc.shape[0])]
            scores = scores + list(acc)
        df = pd.DataFrame({"scores": scores, "names": names})

        curr_ax = ax[i // 3, i % 3]
        sns.boxplot(
            data=df,
            x="names",
            y="scores",
            fill=True,
            gap=0.1,
            color=(0.125, 0.125, 0.875),
            ax=curr_ax,
        )
        curr_ax.set_title(f"{dataset_name}", fontsize=12)
        curr_ax.set_ylabel(None)
        curr_ax.set_xlabel(None)

        num_artists = len(curr_ax.patches)
        num_lines = len(curr_ax.lines)
        lines_per_artist = num_lines // num_artists

        for i, artist in enumerate(curr_ax.patches):
            color = artist.get_facecolor()
            lcolor = _enlighten(color)
            artist.set_color(lcolor)
            artist.set_edgecolor(color)
            for j in range(lines_per_artist):
                curr_ax.lines[i * lines_per_artist + j].set_color(color)

    fig.suptitle(
        f"Accuracies obtained on data subsets selected by different methods on model {model_name}",
        fontsize=16,
    )
    fig.supxlabel("Method")
    fig.supylabel("Accuracy")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.show()


def plot_n_features_ratio_single_dataset(df, ax):
    """Plot outliers ratio for each method for single dataset."""
    true_relevant = df["n_relevant"].reset_index(drop=True)[0]
    bars = ax.bar(
        df["method"],
        df["selected_ratio"],
        color=(0.65625, 0.09375, 0.65625, 0.1),
        edgecolor=(0.65625, 0.09375, 0.65625),
    )
    ax.set_ylim([0, 1])
    if true_relevant is not None:
        ax.axhline(
            y=true_relevant / df["n_features"].reset_index(drop=True)[0],
            color=(0.125, 0.125, 0.875),
            linestyle="--",
            linewidth=1.5,
        )
    for label in ax.get_xticklabels():
        label.set_fontsize(9)
        label.set_ha("right")

    for bar, n_selected in zip(bars, df["n_selected"]):
        height = bar.get_height() + 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{n_selected}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=(0.65625, 0.09375, 0.65625),
        )


def plot_n_features_ratio(results):
    """Plot comparison of outliers ratio for each method and dataset with true outlies ratio."""
    fig, ax = plt.subplots(3, 3, figsize=(12, 12), sharey=True)
    for i, dataset_name in enumerate(np.unique(results["dataset"])):
        plot_n_features_ratio_single_dataset(
            results[results["dataset"] == dataset_name], ax[i // 3, i % 3]
        )
        ax[i // 3, i % 3].set_title(dataset_name)
    fig.suptitle(
        "Comparison of selected features ratio for each method with true relevant ratio (dashed line) for synthetic data",
        fontsize=16,
    )
    fig.supxlabel("Method")
    fig.supylabel("Selected features ratio")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.show()
