"""
Plotting utilities for language modeling experiments.

By Graham Neubig for [11-664/763 Inference Algorithms for Language Modeling]

This module provides reusable plotting functions to create consistent visualizations
across different notebooks and experiments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_heatmap(
    data: np.ndarray | list[float],
    title: str = "",
    cmap: str = "viridis",
    xlabel: str = "",
    ylabel: str = "",
    xticks: Sequence[Any] | None = None,
    yticks: Sequence[Any] | None = None,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Create a heatmap visualization with colorbar.

    Args:
        data: 2D tensor or array to visualize
        title: Title for the plot
        cmap: Colormap to use
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        xticks: X-axis tick positions
        yticks: Y-axis tick positions
        xticklabels: X-axis tick labels
        yticklabels: Y-axis tick labels
        ax: Matplotlib axis to plot on (optional)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(data, cmap=cmap)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Set ticks and labels
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    plt.colorbar(im, ax=ax)

    return ax


def plot_bar_chart(
    data: np.ndarray | list[float],
    labels: Sequence[Any] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str = "lightblue",
    xticks: Sequence[Any] | None = None,
    xticklabels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Create a bar chart visualization.

    Args:
        data: 1D tensor or array of values
        labels: Labels for x-axis (optional)
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        color: Bar color
        xticks: X-axis tick positions
        xticklabels: X-axis tick labels
        ax: Matplotlib axis to plot on (optional)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if labels is None:
        labels = range(len(data))

    ax.bar(labels, data, alpha=0.7, color=color)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Set ticks and labels
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    return ax


def create_subplot_grid(
    rows: int, cols: int, figsize: tuple[int, int] = (12, 8)
) -> tuple[Figure, Any]:
    """
    Create a subplot grid with consistent styling.

    Args:
        rows: Number of rows
        cols: Number of columns
        figsize: Figure size tuple

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        # For multi-dimensional grids, flatten to 1D array
        axes = axes.flatten() if hasattr(axes, "flatten") else axes

    return fig, axes


def plot_probability_distribution(
    probs: np.ndarray,
    labels: Sequence[Any] | None = None,
    title: str = "Probability Distribution",
    temperature: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Plot a probability distribution with optional temperature annotation.

    Args:
        probs: Probability values
        labels: Labels for each probability (optional)
        title: Plot title
        temperature: Temperature value to display (optional)
        ax: Matplotlib axis to plot on (optional)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if labels is None:
        labels = range(len(probs))

    ax.bar(labels, probs, alpha=0.7, color="lightblue")
    ax.set_title(title)
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(probs) * 1.1)

    # Add temperature annotation if provided
    if temperature is not None:
        ax.text(
            0.02,
            0.98,
            f"T = {temperature}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    return ax


def plot_comparison_bars(
    data_dict: dict[str, float],
    title: str = "Comparison",
    ylabel: str = "Value",
    colors: Any = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Create a comparison bar chart from a dictionary of data.

    Args:
        data_dict: Dictionary with labels as keys and values as data
        title: Plot title
        ylabel: Y-axis label
        colors: List of colors for bars (optional)
        ax: Matplotlib axis to plot on (optional)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    labels = list(data_dict.keys())
    values = list(data_dict.values())

    if colors is None:
        colors = plt.cm.get_cmap("Set3")(np.linspace(0, 1, len(labels)))

    bars = ax.bar(labels, values, color=colors, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    return ax


def plot_loglog(
    x_data: np.ndarray,
    y_data: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    label: str = "",
    ax: Axes | None = None,
) -> Axes:
    """
    Create a log-log plot with optional labels and title.

    Args:
        x_data: X-axis data
        y_data: Y-axis data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        label: Data series label
        ax: Matplotlib axis to plot on (optional)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(x_data, y_data, "o-", label=label)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True)

    return ax


def plot_comparison_bars_side_by_side(
    data1: np.ndarray,
    data2: np.ndarray,
    labels1: str = "Data 1",
    labels2: str = "Data 2",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    xticks: Sequence[Any] | None = None,
    xticklabels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Create side-by-side bar chart for comparing two datasets.

    Args:
        data1: First dataset
        data2: Second dataset
        labels1: Label for first dataset
        labels2: Label for second dataset
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        xticks: X-axis tick positions
        xticklabels: X-axis tick labels
        ax: Matplotlib axis to plot on (optional)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    x = range(len(data1))
    width = 0.35

    ax.bar([xi - width / 2 for xi in x], data1, width, label=labels1, alpha=0.7)
    ax.bar([xi + width / 2 for xi in x], data2, width, label=labels2, alpha=0.7)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Set ticks and labels
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    ax.legend()

    return ax


def setup_axis_with_rotation(
    ax: Axes,
    xticks: Sequence[Any] | None = None,
    xticklabels: Sequence[str] | None = None,
    yticks: Sequence[int] | None = None,
    yticklabels: Sequence[str] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    rotation: int = 45,
    fontsize: int = 8,
) -> None:
    """
    Set up axis with common formatting including rotated labels.

    Args:
        ax: Matplotlib axis
        xticks: X-axis tick positions
        xticklabels: X-axis tick labels
        yticks: Y-axis tick positions
        yticklabels: Y-axis tick labels
        title: Axis title
        xlabel: X-axis label
        ylabel: Y-axis label
        rotation: Label rotation angle
        fontsize: Font size for labels
    """
    if title:
        ax.set_title(title, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(
            xticklabels, rotation=rotation, ha="right", fontsize=fontsize
        )
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)


def plot_efficiency_dashboard(
    results_data: dict[str, Any],
    strategies: list[str],
    models: list[str] | None = None,
) -> Figure:
    """
    Create comprehensive efficiency analysis dashboard.

    Args:
        results_data: Dictionary containing efficiency metrics
        strategies: List of strategy names
        models: List of model names (optional)

    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        "Language Model Sampling Efficiency Analysis\n"
        "Comprehensive Performance Metrics Across Different Strategies",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Throughput by strategy
    ax1 = plt.subplot(4, 3, 1)
    throughput_data = results_data.get("throughput", {})
    if throughput_data:
        strategy_throughput = [throughput_data.get(s, 0) for s in strategies]
        colors = plt.cm.get_cmap("Set3")(np.linspace(0, 1, len(strategies)))
        bars = ax1.bar(
            range(len(strategies)), strategy_throughput, color=colors, alpha=0.8
        )

        # Add value labels on bars
        for bar, value in zip(bars, strategy_throughput):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    setup_axis_with_rotation(
        ax1,
        xticks=range(len(strategies)),
        xticklabels=strategies,
        title="Throughput by Strategy",
        ylabel="Tokens/Second",
    )

    # Latency vs Throughput scatter
    ax2 = plt.subplot(4, 3, 2)
    latency_data = results_data.get("latency", {})
    if latency_data and throughput_data:
        for _i, strategy in enumerate(strategies):
            lat = latency_data.get(strategy, 0)
            thr = throughput_data.get(strategy, 0)
            ax2.scatter(lat, thr, s=100, alpha=0.7, label=strategy)

    setup_axis_with_rotation(
        ax2,
        title="Latency vs Throughput Trade-off",
        xlabel="Latency (seconds)",
        ylabel="Throughput (tokens/sec)",
    )
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Memory usage
    ax3 = plt.subplot(4, 3, 3)
    memory_data = results_data.get("memory", {})
    if memory_data:
        strategy_memory = [memory_data.get(s, 0) for s in strategies]
        ax3.bar(range(len(strategies)), strategy_memory, alpha=0.7, color="lightcoral")

    setup_axis_with_rotation(
        ax3,
        xticks=range(len(strategies)),
        xticklabels=strategies,
        title="Memory Usage by Strategy",
        ylabel="Memory (MB)",
    )

    # Parameter sensitivity plots (4-8)
    sensitivity_data = results_data.get("sensitivity", {})

    # Batch size sensitivity
    if "batch_size" in sensitivity_data:
        ax4 = plt.subplot(4, 3, 4)
        batch_data = sensitivity_data["batch_size"]
        for strategy, data in batch_data.items():
            ax4.plot(
                data["batch_sizes"], data["throughput"], "o-", label=strategy, alpha=0.7
            )
        setup_axis_with_rotation(
            ax4,
            title="Batch Size vs Throughput",
            xlabel="Batch Size",
            ylabel="Throughput (tokens/sec)",
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Temperature sensitivity
    if "temperature" in sensitivity_data:
        ax5 = plt.subplot(4, 3, 5)
        temp_data = sensitivity_data["temperature"]
        for strategy, data in temp_data.items():
            ax5.plot(
                data["temperatures"],
                data["throughput"],
                "o-",
                label=strategy,
                alpha=0.7,
            )
        setup_axis_with_rotation(
            ax5,
            title="Temperature Sensitivity",
            xlabel="Temperature",
            ylabel="Throughput (tokens/sec)",
        )
        ax5.grid(True, alpha=0.3)

    # Top-k sensitivity
    if "top_k" in sensitivity_data:
        ax6 = plt.subplot(4, 3, 6)
        topk_data = sensitivity_data["top_k"]
        for strategy, data in topk_data.items():
            ax6.plot(
                data["top_k_values"],
                data["throughput"],
                "o-",
                label=strategy,
                alpha=0.7,
            )
        setup_axis_with_rotation(
            ax6,
            title="Top-k Sensitivity",
            xlabel="Top-k Value",
            ylabel="Throughput (tokens/sec)",
        )
        ax6.grid(True, alpha=0.3)

    # Top-p sensitivity
    if "top_p" in sensitivity_data:
        ax7 = plt.subplot(4, 3, 7)
        topp_data = sensitivity_data["top_p"]
        for strategy, data in topp_data.items():
            ax7.plot(
                data["top_p_values"],
                data["throughput"],
                "o-",
                label=strategy,
                alpha=0.7,
            )
        setup_axis_with_rotation(
            ax7,
            title="Top-p Sensitivity",
            xlabel="Top-p Value",
            ylabel="Throughput (tokens/sec)",
        )
        ax7.grid(True, alpha=0.3)

    # Length scaling
    if "max_length" in sensitivity_data:
        ax8 = plt.subplot(4, 3, 8)
        length_data = sensitivity_data["max_length"]
        for strategy, data in length_data.items():
            ax8.plot(
                data["max_lengths"], data["throughput"], "o-", label=strategy, alpha=0.7
            )
        setup_axis_with_rotation(
            ax8,
            title="Length Scaling",
            xlabel="Max Length",
            ylabel="Throughput (tokens/sec)",
        )
        ax8.grid(True, alpha=0.3)

    # Efficiency heatmap
    ax9 = plt.subplot(4, 3, 9)
    if models and throughput_data:
        # Create heatmap data
        heatmap_data = np.zeros((len(models), len(strategies)))
        for i, model in enumerate(models):
            for j, strategy in enumerate(strategies):
                key = f"{model}_{strategy}"
                heatmap_data[i, j] = throughput_data.get(key, 0)

        im = ax9.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
        setup_axis_with_rotation(
            ax9,
            xticks=range(len(strategies)),
            xticklabels=strategies,
            yticks=range(len(models)),
            yticklabels=models,
            title="Efficiency Heatmap",
        )
        plt.colorbar(im, ax=ax9, label="Throughput (tokens/sec)")

    plt.tight_layout()
    return fig


def plot_generation_dashboard(
    results_data: dict[str, Any],
    strategies: list[str],
    models: list[str] | None = None,
) -> Figure:
    """
    Create comprehensive generation quality analysis dashboard.

    Args:
        results_data: Dictionary containing generation metrics
        strategies: List of strategy names
        models: List of model names (optional)

    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Language Model Generation Quality Analysis\n"
        "Comprehensive Quality Metrics Across Different Strategies",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Average text length by strategy
    ax1 = plt.subplot(3, 3, 1)
    length_data = results_data.get("avg_length", {})
    if length_data:
        plot_comparison_bars(
            length_data,
            title="Average Text Length by Strategy",
            ylabel="Average Length (words)",
            ax=ax1,
        )
        setup_axis_with_rotation(
            ax1, xticks=range(len(strategies)), xticklabels=strategies
        )

    # Generation time by strategy
    ax2 = plt.subplot(3, 3, 2)
    time_data = results_data.get("avg_time", {})
    if time_data:
        plot_comparison_bars(
            time_data,
            title="Average Generation Time by Strategy",
            ylabel="Average Time (seconds)",
            ax=ax2,
        )
        setup_axis_with_rotation(
            ax2, xticks=range(len(strategies)), xticklabels=strategies
        )

    # Text diversity by strategy
    ax3 = plt.subplot(3, 3, 3)
    diversity_data = results_data.get("diversity", {})
    if diversity_data:
        plot_comparison_bars(
            diversity_data,
            title="Text Diversity by Strategy",
            ylabel="Unique Words Ratio (%)",
            ax=ax3,
        )
        setup_axis_with_rotation(
            ax3, xticks=range(len(strategies)), xticklabels=strategies
        )

    # Average text length by model
    ax4 = plt.subplot(3, 3, 4)
    if models and length_data:
        model_length_data = {model: length_data.get(model, 0) for model in models}
        plot_comparison_bars(
            model_length_data,
            title="Average Text Length by Model",
            ylabel="Average Length (words)",
            ax=ax4,
        )
        setup_axis_with_rotation(ax4, xticks=range(len(models)), xticklabels=models)

    # Fluency scores (if available)
    fluency_data = results_data.get("fluency", {})
    if fluency_data:
        ax5 = plt.subplot(3, 3, 5)
        plot_comparison_bars(
            fluency_data,
            title="Average Fluency Score by Strategy",
            ylabel="Fluency Score (1-10)",
            ax=ax5,
        )
        setup_axis_with_rotation(
            ax5, xticks=range(len(strategies)), xticklabels=strategies
        )
        ax5.set_ylim(0, 10)

    # Speed vs Diversity tradeoff
    ax6 = plt.subplot(3, 3, 6)
    if time_data and diversity_data:
        for strategy in strategies:
            time_val = time_data.get(strategy, 0)
            div_val = diversity_data.get(strategy, 0)
            ax6.scatter(time_val, div_val, s=100, alpha=0.7, label=strategy)

    setup_axis_with_rotation(
        ax6,
        title="Speed vs Diversity Tradeoff",
        xlabel="Average Generation Time (s)",
        ylabel="Unique Words Ratio",
    )
    ax6.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Generation throughput (if available)
    throughput_data = results_data.get("throughput", {})
    if throughput_data:
        ax7 = plt.subplot(3, 3, 7)
        plot_comparison_bars(
            throughput_data,
            title="Generation Throughput",
            ylabel="Tokens per Second",
            ax=ax7,
        )
        setup_axis_with_rotation(
            ax7, xticks=range(len(strategies)), xticklabels=strategies
        )

    plt.tight_layout()
    return fig


def plot_generation_incremental(
    results_data: dict[str, Any],
    strategies: list[str],
    models: list[str] | None = None,
    plot_type: str = "length",
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """
    Create individual generation quality plots for incremental display.

    Args:
        results_data: Dictionary containing generation metrics
        strategies: List of strategy names
        models: List of model names (optional)
        plot_type: Type of plot to create ('length', 'time', 'diversity', 'fluency', 'tradeoff', 'throughput')
        figsize: Figure size tuple

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "length":
        length_data = results_data.get("avg_length", {})
        if length_data:
            plot_comparison_bars(
                length_data,
                title="Average Text Length by Strategy",
                ylabel="Average Length (words)",
                ax=ax,
            )
            setup_axis_with_rotation(
                ax, xticks=range(len(strategies)), xticklabels=strategies
            )

    elif plot_type == "time":
        time_data = results_data.get("avg_time", {})
        if time_data:
            plot_comparison_bars(
                time_data,
                title="Average Generation Time by Strategy",
                ylabel="Average Time (seconds)",
                ax=ax,
            )
            setup_axis_with_rotation(
                ax, xticks=range(len(strategies)), xticklabels=strategies
            )

    elif plot_type == "diversity":
        diversity_data = results_data.get("diversity", {})
        if diversity_data:
            plot_comparison_bars(
                diversity_data,
                title="Text Diversity by Strategy",
                ylabel="Unique Words Ratio (%)",
                ax=ax,
            )
            setup_axis_with_rotation(
                ax, xticks=range(len(strategies)), xticklabels=strategies
            )

    elif plot_type == "fluency":
        fluency_data = results_data.get("fluency", {})
        if fluency_data:
            plot_comparison_bars(
                fluency_data,
                title="Average Fluency Score by Strategy",
                ylabel="Fluency Score (1-10)",
                ax=ax,
            )
            setup_axis_with_rotation(
                ax, xticks=range(len(strategies)), xticklabels=strategies
            )
            ax.set_ylim(0, 10)

    elif plot_type == "tradeoff":
        time_data = results_data.get("avg_time", {})
        diversity_data = results_data.get("diversity", {})
        if time_data and diversity_data:
            colors = plt.cm.get_cmap("Set3")(np.linspace(0, 1, len(strategies)))
            for i, strategy in enumerate(strategies):
                time_val = time_data.get(strategy, 0)
                div_val = diversity_data.get(strategy, 0)
                ax.scatter(
                    time_val, div_val, s=100, alpha=0.7, label=strategy, color=colors[i]
                )

        setup_axis_with_rotation(
            ax,
            title="Speed vs Diversity Tradeoff",
            xlabel="Average Generation Time (s)",
            ylabel="Unique Words Ratio (%)",
        )
        ax.legend()

    elif plot_type == "throughput":
        throughput_data = results_data.get("throughput", {})
        if throughput_data:
            plot_comparison_bars(
                throughput_data,
                title="Generation Throughput",
                ylabel="Tokens per Second",
                ax=ax,
            )
            setup_axis_with_rotation(
                ax, xticks=range(len(strategies)), xticklabels=strategies
            )

    elif plot_type == "model_comparison" and models:
        length_data = results_data.get("avg_length", {})
        if length_data:
            model_length_data = {model: length_data.get(model, 0) for model in models}
            plot_comparison_bars(
                model_length_data,
                title="Average Text Length by Model",
                ylabel="Average Length (words)",
                ax=ax,
            )
            setup_axis_with_rotation(ax, xticks=range(len(models)), xticklabels=models)

    plt.tight_layout()
    return fig


def plot_meta_generation_dashboard(
    results_data: dict[str, Any],
    generators: list[str],
) -> Figure:
    """
    Create comprehensive meta-generation analysis dashboard.

    Args:
        results_data: Dictionary containing meta-generation metrics
        generators: List of generator names

    Returns:
        fig: Matplotlib figure
    """
    plt.style.use("default")
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Meta-Generation Analysis: Model Preference Evaluation\n"
        "Comparing Different Generators Using Multiple Evaluator Models",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Likelihood ratio distribution
    ax1 = plt.subplot(3, 3, 1)
    likelihood_data = results_data.get("likelihood_ratios", [])
    if likelihood_data:
        ax1.hist(
            likelihood_data, bins=30, alpha=0.7, color="skyblue", label="Medium - Small"
        )
        ax1.axvline(0, color="red", linestyle="--", alpha=0.7, label="No Preference")
        setup_axis_with_rotation(
            ax1,
            title="Distribution of Model Preferences",
            xlabel="Likelihood Ratio (Medium - Small)",
            ylabel="Frequency",
        )
        ax1.legend()

    # Preference by generator
    ax2 = plt.subplot(3, 3, 2)
    preference_data = results_data.get("preferences_by_generator", {})
    if preference_data:
        x = np.arange(len(generators))
        small_prefs = [
            preference_data.get(gen, {}).get("small", 0) for gen in generators
        ]
        medium_prefs = [
            preference_data.get(gen, {}).get("medium", 0) for gen in generators
        ]

        width = 0.35
        ax2.bar(
            x - width / 2, small_prefs, width, label="Small Model Preferred", alpha=0.7
        )
        ax2.bar(
            x + width / 2,
            medium_prefs,
            width,
            label="Medium Model Preferred",
            alpha=0.7,
        )

        setup_axis_with_rotation(
            ax2,
            xticks=x.tolist(),
            xticklabels=generators,
            title="Preference by Generator",
            ylabel="Number of Texts",
        )
        ax2.legend()

    # Length vs preference
    ax3 = plt.subplot(3, 3, 3)
    length_pref_data = results_data.get("length_vs_preference", {})
    if length_pref_data:
        lengths = length_pref_data.get("lengths", [])
        ratios = length_pref_data.get("ratios", [])
        ax3.scatter(lengths, ratios, alpha=0.6)
        ax3.axhline(0, color="red", linestyle="--", alpha=0.7)
        setup_axis_with_rotation(
            ax3,
            title="Length vs Model Preference",
            xlabel="Text Length (words)",
            ylabel="Likelihood Ratio",
        )

    # Perplexity comparison
    ax4 = plt.subplot(3, 3, 4)
    perplexity_data = results_data.get("perplexity_comparison", {})
    if perplexity_data:
        small_perp = perplexity_data.get("small", [])
        medium_perp = perplexity_data.get("medium", [])
        ax4.scatter(small_perp, medium_perp, alpha=0.6, label="Generated Texts")

        # Add diagonal line
        min_val = min(min(small_perp), min(medium_perp))
        max_val = max(max(small_perp), max(medium_perp))
        ax4.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            alpha=0.7,
            label="Equal Perplexity",
        )

        setup_axis_with_rotation(
            ax4,
            title="Perplexity Comparison",
            xlabel="Small Model Perplexity",
            ylabel="Medium Model Perplexity",
        )
        ax4.legend()

    # Generation speed vs preference
    ax5 = plt.subplot(3, 3, 5)
    speed_pref_data = results_data.get("speed_vs_preference", {})
    if speed_pref_data:
        speeds = speed_pref_data.get("speeds", [])
        ratios = speed_pref_data.get("ratios", [])
        ax5.scatter(speeds, ratios, alpha=0.6)
        ax5.axhline(0, color="red", linestyle="--", alpha=0.7)
        setup_axis_with_rotation(
            ax5,
            title="Generation Speed vs Preference",
            xlabel="Generation Time (seconds)",
            ylabel="Likelihood Ratio",
        )

    # Preference distribution by generator
    ax6 = plt.subplot(3, 3, 6)
    if preference_data:
        for _i, gen in enumerate(generators):
            gen_ratios = results_data.get("ratios_by_generator", {}).get(gen, [])
            if gen_ratios:
                ax6.hist(gen_ratios, bins=20, alpha=0.5, label=gen)

        setup_axis_with_rotation(
            ax6, title="Preference Distribution by Generator", ylabel="Likelihood Ratio"
        )
        ax6.legend()

    # Likelihood heatmap
    ax7 = plt.subplot(3, 3, 7)
    heatmap_data = results_data.get("likelihood_heatmap")
    if heatmap_data is not None:
        im = ax7.imshow(heatmap_data, cmap="RdBu_r", aspect="auto")
        setup_axis_with_rotation(
            ax7,
            xticks=range(len(generators)),
            xticklabels=generators,
            yticks=[0, 1],
            yticklabels=["Small Model", "Medium Model"],
            title="Average Likelihood Heatmap",
        )
        plt.colorbar(im, ax=ax7, label="Average Log Likelihood")

    plt.tight_layout()
    return fig


def plot_conditional_probability_heatmap(
    data: np.ndarray,
    vocab: list[str],
    title: str = "",
    xlabel: str = "Next Word",
    ylabel: str = "Current Word",
    ax: Axes | None = None,
) -> Axes:
    """
    Plot conditional probability heatmap with proper axis labels.

    Args:
        data: 2D probability matrix
        vocab: Vocabulary list for labels
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    plot_heatmap(data, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
    setup_axis_with_rotation(
        ax,
        xticks=range(len(vocab)),
        xticklabels=vocab,
        yticks=range(len(vocab)),
        yticklabels=vocab,
    )
    return ax


def plot_sampling_comparison(
    original_probs: np.ndarray,
    modified_probs_list: list[np.ndarray],
    labels: list[str],
    focused_words: list[str],
    temperatures: list[float] | None = None,
    top_k_values: list[int] | None = None,
) -> Figure:
    """
    Create comprehensive sampling method comparison plot.

    Args:
        original_probs: Original probability distribution
        modified_probs_list: List of modified probability distributions
        labels: Labels for each distribution
        focused_words: Word labels for x-axis
        temperatures: Temperature values (optional)
        top_k_values: Top-k values (optional)

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = create_subplot_grid(2, 4, figsize=(16, 8))

    # Original distribution
    plot_bar_chart(
        original_probs,
        title="Original P(word|'the')",
        xlabel="Next Word",
        ylabel="Probability",
        ax=axes[0],
    )
    setup_axis_with_rotation(
        axes[0], xticks=range(len(focused_words)), xticklabels=focused_words
    )

    # Temperature variations
    if temperatures:
        for i, (temp, probs) in enumerate(zip(temperatures, modified_probs_list[:3])):
            plot_bar_chart(
                probs,
                title=f"Temperature = {temp}",
                xlabel="Next Word",
                ylabel="Probability",
                ax=axes[i + 1],
            )
            setup_axis_with_rotation(
                axes[i + 1], xticks=range(len(focused_words)), xticklabels=focused_words
            )

    # Top-k variations
    if top_k_values:
        start_idx = 4 if temperatures else 1
        for i, (k, probs) in enumerate(zip(top_k_values, modified_probs_list[3:6])):
            plot_bar_chart(
                probs,
                title=f"Top-k (k={k})",
                xlabel="Next Word",
                ylabel="Probability",
                ax=axes[start_idx + i],
            )
            setup_axis_with_rotation(
                axes[start_idx + i],
                xticks=range(len(focused_words)),
                xticklabels=focused_words,
            )

    # Nucleus sampling
    if len(modified_probs_list) > 6:
        plot_bar_chart(
            modified_probs_list[-1],
            title="Nucleus (p=0.8)",
            xlabel="Next Word",
            ylabel="Probability",
            ax=axes[7],
        )
        setup_axis_with_rotation(
            axes[7], xticks=range(len(focused_words)), xticklabels=focused_words
        )

    plt.tight_layout()
    return fig


def plot_convergence_analysis(
    convergence_data: dict[str, dict[str, list[float]]],
    temperatures: list[float],
) -> Figure:
    """
    Plot convergence analysis for different temperatures.

    Args:
        convergence_data: Dictionary with convergence data for each temperature
        temperatures: List of temperature values

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = create_subplot_grid(1, 3, figsize=(15, 4))

    for i, temp in enumerate(temperatures):
        data = convergence_data.get(str(temp), {})
        sample_counts = data.get("sample_counts", [])
        errors = data.get("errors", [])

        if sample_counts and errors:
            axes[i].plot(sample_counts, errors, "b-", alpha=0.7)
            setup_axis_with_rotation(
                axes[i],
                title=f"Convergence at T={temp}",
                xlabel="Number of Samples",
                ylabel="Mean Absolute Error",
            )

    plt.tight_layout()
    return fig


def plot_mle_analysis(
    true_probs: np.ndarray,
    mle_probs: list[np.ndarray],
    sample_sizes: list[int],
    errors: list[float],
    key_words: list[str],
) -> Figure:
    """
    Plot Maximum Likelihood Estimation analysis.

    Args:
        true_probs: True probability distribution
        mle_probs: MLE estimated probabilities
        sample_sizes: List of sample sizes used
        errors: Corresponding errors for each sample size
        key_words: Word labels

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = create_subplot_grid(1, 2, figsize=(12, 4))

    # True vs MLE comparison
    x = range(len(key_words))
    width = 0.35
    axes[0].bar(
        [xi - width / 2 for xi in x], true_probs, width, label="True", alpha=0.7
    )
    axes[0].bar([xi + width / 2 for xi in x], mle_probs, width, label="MLE", alpha=0.7)

    setup_axis_with_rotation(
        axes[0],
        xticks=x,
        xticklabels=key_words,
        title="True vs MLE Probabilities",
        ylabel="Probability",
    )
    axes[0].legend()

    # Error vs sample size
    axes[1].plot(sample_sizes, errors, "bo-", alpha=0.7)
    setup_axis_with_rotation(
        axes[1],
        title="MLE Error vs Sample Size",
        xlabel="Number of Samples",
        ylabel="Mean Absolute Error",
    )

    plt.tight_layout()
    return fig


def save_figure(
    fig: Figure,
    filename_base: str,
    dpi: int = 300,
) -> None:
    """
    Save figure in multiple formats with consistent settings.

    Args:
        fig: Matplotlib figure
        filename_base: Base filename (without extension)
        dpi: Resolution for PNG output
    """
    fig.savefig(f"{filename_base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{filename_base}.pdf", bbox_inches="tight")
