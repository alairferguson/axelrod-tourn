"""Shared matplotlib styling for publication / blog figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch

# Light, GitHub-adjacent palette — reads well on blog posts and slides.
BG = "#FAFBFC"
FG = "#1F2328"
MUTED = "#656D76"
GRID = "#D8DEE4"
SPINE = "#D0D7DE"
CLASSIC = "#8B949E"
LLM_ACCENTS = ("#218BFF", "#A371F7", "#FF7B72", "#3FB950", "#D29922", "#F778BA")


def apply_theme() -> None:
    """Set global rcParams for a clean, modern sans-serif look."""
    rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": FG,
            "axes.titlecolor": FG,
            "text.color": FG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Helvetica Neue",
                "Segoe UI",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "600",
            "axes.labelsize": 12,
            "axes.labelweight": "500",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "savefig.facecolor": BG,
            "savefig.dpi": 200,
        }
    )


def style_axes(ax, *, xlabel: str, ylabel: str) -> None:
    """Apply shared axis cosmetics after plotting."""
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(True, alpha=0.55, linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(SPINE)
        ax.spines[spine].set_linewidth(0.8)


def add_title_block(
    fig,
    title: str,
    subtitle: str | None = None,
    *,
    left: float = 0.09,
) -> None:
    """Figure-level title + subtitle with fixed vertical spacing."""
    fig.text(
        left, 0.97, title,
        fontsize=16, fontweight="600", color=FG, ha="left", va="top",
    )
    if subtitle:
        fig.text(
            left, 0.932, subtitle,
            fontsize=11, color=MUTED, ha="left", va="top",
        )


def marker_size(retaliation: float, *, llm: bool) -> float:
    """Map retaliation [0, 1] to a readable scatter area."""
    lo, hi = (140, 420) if llm else (70, 260)
    r = max(0.0, min(1.0, retaliation)) if retaliation == retaliation else 0.5
    return lo + r * (hi - lo)


def draw_fingerprint_legend(fig, *, left: float = 0.09, bottom: float = 0.028) -> None:
    """Compact horizontal legend below the plot — player types + retaliation scale."""
    overlay = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=10)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")

    panel_left = left
    panel_w = 0.348
    panel_h = 0.052
    row_y = bottom + 0.026

    def _fig_text(x, y, text, *, weight="400", color=MUTED, size=8.5, ha="left"):
        overlay.text(
            x, y, text, fontsize=size, fontweight=weight, color=color,
            ha=ha, va="center", transform=fig.transFigure, zorder=3,
        )

    def _marker(x, y, *, size, color, marker="o"):
        overlay.scatter(
            [x], [y], s=size, c=color, marker=marker,
            edgecolors="white", linewidths=1.2,
            transform=fig.transFigure, zorder=3, clip_on=False,
        )

    # Player types
    classic_x = panel_left + 0.012
    _marker(classic_x, row_y, size=68, color=CLASSIC)
    _fig_text(classic_x + 0.009, row_y, "Classic strategy", color=FG)

    llm_x = panel_left + 0.112
    _marker(llm_x, row_y, size=68, color=LLM_ACCENTS[0], marker="D")
    _fig_text(llm_x + 0.009, row_y, "LLM player", color=FG)

    div_x = panel_left + 0.198
    overlay.plot(
        [div_x, div_x], [bottom + 0.003, bottom + panel_h - 0.006],
        color=GRID, linewidth=1.0, transform=fig.transFigure, zorder=2,
    )

    # Retaliation scale — same dot spacing as before (0.10 figure units)
    low_x, high_x = panel_left + 0.218, panel_left + 0.318
    scale_y = row_y - 0.001
    scale_center = (low_x + high_x) / 2
    dot_xs = [low_x + i * (high_x - low_x) / 4 for i in range(5)]
    dot_sizes = [22, 52, 88, 128, 168]

    overlay.text(
        scale_center, row_y + 0.013, "Retaliation",
        fontsize=8.5, fontweight="500", color=FG,
        ha="center", va="center", transform=fig.transFigure, zorder=3,
    )
    overlay.scatter(
        dot_xs, [scale_y] * 5, s=dot_sizes, c=CLASSIC,
        marker="o", edgecolors="white", linewidths=1.0,
        transform=fig.transFigure, zorder=3, clip_on=False,
    )

    tick_y = bottom + 0.003
    for x, label in ((low_x, "0.0"), (scale_center, "0.5"), (high_x, "1.0")):
        _fig_text(x, tick_y, label, size=7, ha="center")

    panel = FancyBboxPatch(
        (panel_left - 0.006, bottom - 0.002), panel_w + 0.012, panel_h,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        transform=fig.transFigure,
        facecolor="#FFFFFF",
        edgecolor=GRID,
        linewidth=1.0,
        alpha=0.98,
        zorder=1,
        clip_on=False,
    )
    overlay.add_patch(panel)
