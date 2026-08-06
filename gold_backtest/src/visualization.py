"""Charts: equity curves, example trade annotations, comparison plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .backtest_engine import BacktestResult, Trade
from .strategy_pmts import Signal

RESULTS = Path(__file__).resolve().parent.parent / "results"
CHARTS = RESULTS / "charts"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f1419",
            "axes.facecolor": "#1a2332",
            "axes.edgecolor": "#4a5568",
            "axes.labelcolor": "#e2e8f0",
            "text.color": "#e2e8f0",
            "xtick.color": "#a0aec0",
            "ytick.color": "#a0aec0",
            "grid.color": "#2d3748",
            "grid.alpha": 0.6,
            "font.family": "DejaVu Sans",
        }
    )


def plot_equity(
    results: dict[str, BacktestResult],
    title: str,
    path: Path,
    is_end: pd.Timestamp | None = None,
) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#38bdf8", "#fbbf24", "#34d399", "#f472b6", "#a78bfa"]
    for i, (name, res) in enumerate(results.items()):
        eq = res.equity.sort_index()
        # Normalize to start at 100
        norm = 100.0 * eq / eq.iloc[0]
        ax.plot(norm.index, norm.values, label=name, color=colors[i % len(colors)], lw=1.8)
    if is_end is not None:
        ax.axvline(is_end, color="#fc8181", ls="--", lw=1.2, label="IS → OOS split")
    ax.set_title(title)
    ax.set_ylabel("Equity (normalized=100)")
    ax.legend(loc="best", facecolor="#1a2332", edgecolor="#4a5568")
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_example_trade(
    df_ltf: pd.DataFrame,
    trade: Trade,
    signal: Signal | None,
    path: Path,
    pad_bars: int = 40,
    bar_minutes: int = 15,
) -> Path:
    """Candlestick-ish OHLC plot with entry/exit/zone annotations."""
    _style()
    df = df_ltf.copy()
    t0 = trade.entry_time - pd.Timedelta(minutes=bar_minutes * pad_bars)
    t1 = trade.exit_time + pd.Timedelta(minutes=bar_minutes * pad_bars)
    win = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)].reset_index(drop=True)
    if win.empty:
        return path

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(win))
    for i, row in win.iterrows():
        color = "#34d399" if row["close"] >= row["open"] else "#f87171"
        ax.plot([i, i], [row["low"], row["high"]], color=color, lw=0.8)
        ax.plot([i, i], [row["open"], row["close"]], color=color, lw=2.4)

    # Map times to x
    def x_of(ts):
        m = win["timestamp"] == pd.Timestamp(ts)
        if m.any():
            return int(np.where(m.to_numpy())[0][0])
        # nearest
        idx = (win["timestamp"] - pd.Timestamp(ts)).abs().argmin()
        return int(idx)

    xe = x_of(trade.entry_time)
    xx = x_of(trade.exit_time)
    ax.scatter([xe], [trade.entry_price], color="#38bdf8", s=80, zorder=5, label="Entry")
    ax.scatter([xx], [trade.exit_price], color="#fbbf24", s=80, zorder=5, label="Exit")
    ax.axhline(trade.stop, color="#f87171", ls="--", lw=1, label="Stop")
    ax.axhline(trade.take_profit, color="#34d399", ls="--", lw=1, label="TP")

    if signal is not None:
        ax.axhspan(
            signal.setup.zone_bottom,
            signal.setup.zone_top,
            color="#a78bfa",
            alpha=0.18,
            label="Golden zone",
        )
        ax.axhline(signal.setup.ob_low, color="#94a3b8", ls=":", lw=0.9, alpha=0.8)
        ax.axhline(signal.setup.ob_high, color="#94a3b8", ls=":", lw=0.9, alpha=0.8)

    ax.set_title(
        f"{trade.direction.upper()} {trade.pattern or ''} | "
        f"PnL=${trade.pnl:.2f} ({trade.return_R:.2f}R) | {trade.exit_reason}"
    )
    ax.set_ylabel("Price (USDT)")
    step = max(len(win) // 8, 1)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(
        [pd.Timestamp(t).strftime("%m-%d %H:%M") for t in win["timestamp"].iloc[::step]],
        rotation=30,
        ha="right",
    )
    ax.legend(loc="best", facecolor="#1a2332", edgecolor="#4a5568", fontsize=8)
    ax.grid(True, alpha=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_comparison_table_image(rows: list[dict], path: Path, title: str) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(12, 2 + 0.45 * len(rows)))
    ax.axis("off")
    cols = list(rows[0].keys()) if rows else []
    cell = [[str(r[c]) for c in cols] for r in rows]
    table = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    for (r, c), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor("#4a5568")
        if r == 0:
            cell_obj.set_facecolor("#2d3748")
            cell_obj.set_text_props(color="#e2e8f0", weight="bold")
        else:
            cell_obj.set_facecolor("#1a2332")
            cell_obj.set_text_props(color="#e2e8f0")
    ax.set_title(title, pad=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
