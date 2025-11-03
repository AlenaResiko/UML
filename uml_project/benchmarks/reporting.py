"""Summary and plotting helpers for benchmark results."""

from __future__ import annotations

from pathlib import Path
import typing as t

import matplotlib.pyplot as plt
import seaborn as sns

from .utils import MetricResult, ensure_path


def metrics_to_dataframe(metrics: t.Sequence[MetricResult]):
    import pandas as pd

    records = []
    for metric in metrics:
        row = {"metric": metric.name, "value": metric.value}
        if metric.metadata:
            row.update(metric.metadata)
        records.append(row)
    return pd.DataFrame.from_records(records)


def plot_metric_bars(
    metrics: t.Sequence[MetricResult],
    *,
    metric_name: str,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Axes:
    df = metrics_to_dataframe(metrics)
    subset = df[df["metric"] == metric_name]
    if subset.empty:
        raise ValueError(f"No metric named '{metric_name}' present in results")

    ax = sns.barplot(data=subset, x="domain", y="value")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Domain")
    if title:
        ax.set_title(title)

    if save_path is not None:
        path = ensure_path(Path(save_path).parent) / Path(save_path).name
        ax.figure.savefig(path, bbox_inches="tight")

    return ax


__all__ = [
    "metrics_to_dataframe",
    "plot_metric_bars",
]

