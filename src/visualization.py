"""Module for visualization utils."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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


def plot_multiple_plots(accuracy_scores):
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

    fig.delaxes(ax[2, 2])
    fig.suptitle(
        f"Accuracies obtained on data subsets selected by different methods",
        fontsize=16,
    )
    fig.supxlabel("Method")
    fig.supylabel("Accuracy")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.show()
