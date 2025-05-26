import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


def assign_colors(df: pl.DataFrame, colors: sns.color_palette) -> pl.DataFrame:
    """
    Assigns a unique color to each row based on a color palette.
    Extends the palette if there are more rows than colors.
    """
    if len(colors) < len(df):
        repeat_factor = math.ceil(len(df) / len(colors))
        colors = np.tile(colors, (repeat_factor, 1))

    color_series = pl.Series("color", colors[: len(df)])
    return df.insert_column(1, color_series)


def add_bar_labels(ax: plt.Axes) -> None:
    """
    Adds labels to bars in the plot:
    - Inside the bar if the bar is wide enough.
    - Outside the bar otherwise.
    """
    _, legend_labels = ax.get_legend_handles_labels()
    x_range = ax.get_xlim()
    width_threshold = (x_range[1] - x_range[0]) * 0.4

    for container, label in zip(ax.containers, legend_labels):
        for bar in container:
            width = bar.get_width()
            label_text = f"{label}:  {width:.2f}" if width > 0 else ""
            label_pos = "center" if width > width_threshold else "edge"
            padding = 3 if label_pos == "center" else 20
            ax.bar_label(
                container,
                labels=[label_text],
                label_type=label_pos,
                color="black",
                fontsize=18,
                padding=padding,
            )


def create_plot(df: pl.DataFrame, metric_name: str, dataset_name: str, path: Path) -> None:
    """Creates and saves a horizontal bar plot for the given metric."""
    # Reorder based on score but keep colors identical across plots
    df = df.sort(metric_name, descending=True)

    if "color" not in df.columns:
        assign_colors(df, sns.color_palette("tab20"))

    palette = df["color"].to_list()

    # Create figure and plot
    plt.figure(figsize=(17, 10))
    ax = sns.barplot(
        df.drop("color"),
        x=metric_name,
        y="dataset",
        hue="model",
        palette=palette,
        edgecolor="white",
        linewidth=1.5,
    )

    # Set title and labels
    ax.set_title(f"{metric_name.capitalize()} on {dataset_name} Testset", fontsize=24, pad=20)
    ax.set_ylabel("Model", fontsize=20, labelpad=20)
    unit = df[metric_name + "_unit"].unique()[0]
    ax.set_xlabel(f"{metric_name.capitalize()} [{unit}]", fontsize=20, labelpad=25)
    ax.get_legend().remove()

    # Axis and grid formatting
    ax.grid(axis="x")
    ax.tick_params(axis="x", labelsize=18)
    ax.set_axisbelow(True)
    ax.set(xlim=(0, 100 if unit == "%" else df[metric_name].max()), yticklabels=[])
    sns.despine()

    # Add labels to bars
    add_bar_labels(ax)

    fig = ax.get_figure()
    fig.savefig(path)
    return df, fig


def visualize_results(folder: Path, df: pl.DataFrame, metrics: List[str]) -> None:
    """Generates and saves a plot for each metric in the metrics list."""
    path = folder / "plots"
    path.mkdir(parents=True, exist_ok=True)

    # Generate and save plots for each metric and dataset
    for metric in metrics:
        for dataset, sub_df in df.group_by("dataset"):
            dataset_name = dataset[0]
            plot_path = path / f"{metric.capitalize()}_{dataset_name}.png"
            df, _ = create_plot(sub_df, metric, dataset_name, plot_path)
