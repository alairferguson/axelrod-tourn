"""Redesigned report figures — fingerprint heatmap + leaderboard dumbbell.

Reads results/data/*.csv directly (no re-run of the tournament needed) and
writes two new PNGs into results/figures/, alongside the originals:

    fingerprint_heatmap.png    the five-dimension profile as an ink "scan"
    leaderboard_dumbbell.png   rank by wins vs. rank by mean payoff/turn

Run:
    python scripts/redesign_figures.py
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from llm_ipd.plot_style import (
    BG, FG, MUTED, GRID, SPINE, CLASSIC, LLM_ACCENTS,
    apply_theme, add_title_block, blend_hex, build_llm_colors,
)
from llm_ipd.llm_player import parse_llm_player, display_llm_name, llm_player_name

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "results", "data")
FIGURES = os.path.join(ROOT, "results", "figures")

PERSONA_ORDER = ("cooperative", "neutral", "payoff_only", "selfish")
HEAT_LOW = "#E9EBEE"   # visible against BG at value == 0
HEAT_HIGH = FG          # ink, at value == 1


# --- data loading --------------------------------------------------------

def load_fingerprints():
    rows, order = {}, []
    with open(os.path.join(DATA, "fingerprints.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["player"]
            order.append(name)
            rows[name] = {k: float(v) for k, v in row.items() if k != "player"}
    return rows, order


def load_cooperation_matrix():
    path = os.path.join(DATA, "cooperation_matrix.csv")
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)[1:]
        matrix = {}
        for row in reader:
            name = row[0]
            matrix[name] = {col: float(v) for col, v in zip(header, row[1:])}
    return matrix


def load_sample_match(interactions_csv, me, opp, repetition="0"):
    with open(interactions_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    mine_row = next(r for r in rows if r["Player name"] == me
                     and r["Opponent name"] == opp and r["Repetition"] == repetition)
    theirs_row = next(r for r in rows if r["Player name"] == opp
                        and r["Opponent name"] == me and r["Repetition"] == repetition)
    return {
        "mine": mine_row["Actions"],
        "theirs": theirs_row["Actions"],
        "score_per_turn": float(mine_row["Score per turn"]),
        "total_score": float(mine_row["Score"]),
        "turns": int(mine_row["Turns"]),
    }


def load_outcomes():
    rows = {}
    with open(os.path.join(DATA, "outcomes.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["player"]] = {
                "wins": int(row["wins"]),
                "mean_score_per_turn": float(row["mean_score_per_turn"]),
                "total_score": float(row["total_score"]),
                "matches": int(row["matches"]),
            }
    return rows


def competition_ranks(values: dict) -> dict:
    """1 = best. Ties share a rank; the next rank skips (1,1,1,4 not 1,1,1,2)."""
    ranked = {}
    for name, val in values.items():
        better = sum(1 for v2 in values.values() if v2 > val)
        ranked[name] = 1 + better
    return ranked


def model_base_colors(llm_names: list[str]) -> dict:
    """One identity swatch per model, taken from its neutral-persona hue."""
    colors = build_llm_colors(llm_names)
    bases = {}
    for name in llm_names:
        parsed = parse_llm_player(name)
        if parsed and parsed[1] == "neutral":
            bases[parsed[0]] = colors[name]
    return bases


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def _short_name(name: str) -> str:
    parsed = parse_llm_player(name)
    if parsed:
        model, persona = parsed
        return f"{model} · {persona}"
    return name


# --- figure 1: fingerprint heatmap ----------------------------------------

COLUMNS = [
    ("cooperation_rate", "Coop."),
    ("niceness", "Niceness"),
    ("forgiveness", "Forgive."),
    ("retaliation", "Retaliate"),
    ("provocability", "Provoc."),
]


def ordered_rows(fp_order, fingerprints):
    classics = [n for n in fp_order if parse_llm_player(n) is None]
    llms = [n for n in fp_order if parse_llm_player(n) is not None]

    classics_sorted = sorted(
        classics, key=lambda n: -fingerprints[n]["cooperation_rate"]
    )

    models = []
    for n in llms:
        m = parse_llm_player(n)[0]
        if m not in models:
            models.append(m)

    llms_sorted = []
    for m in models:
        for persona in PERSONA_ORDER:
            for n in llms:
                if parse_llm_player(n) == (m, persona):
                    llms_sorted.append(n)
    return classics_sorted, llms_sorted


def draw_fingerprint_heatmap(fingerprints, fp_order, path):
    classics, llms = ordered_rows(fp_order, fingerprints)
    bases = model_base_colors(llms)
    models = list(bases.keys())

    # --- column x positions: a wider gap splits dovish vs. hawkish traits
    col_x = []
    x = 0.0
    for i, _ in enumerate(COLUMNS):
        if i == 3:
            x += 0.55
        col_x.append(x)
        x += 1.0

    # --- row y positions: classics block, gap, then one sub-block per model
    row_y = {}
    y = 0.0
    for n in classics:
        row_y[n] = y
        y += 1.0
    y += 1.4
    block_gap_y = y - 0.95  # divider between classics and LLM blocks

    model_dividers = []
    prev_model = None
    for n in llms:
        m = parse_llm_player(n)[0]
        if prev_model is not None and m != prev_model:
            y += 0.35
            model_dividers.append(y - 0.175)
        row_y[n] = y
        y += 1.0
        prev_model = m

    rows = classics + llms
    total_y = y

    fig = plt.figure(figsize=(9.6, 8.6))
    ax = fig.add_axes([0.30, 0.155, 0.66, 0.70])
    ax.set_facecolor(BG)
    ax.axis("off")

    margin = 0.08
    for n in rows:
        ry = row_y[n]
        for j, (key, _) in enumerate(COLUMNS):
            v = max(0.0, min(1.0, fingerprints[n][key]))
            cx = col_x[j]
            color = blend_hex(HEAT_LOW, HEAT_HIGH, v)
            rect = FancyBboxPatch(
                (cx + margin, ry + margin), 1 - 2 * margin, 1 - 2 * margin,
                boxstyle="round,pad=0,rounding_size=0.10",
                facecolor=color, edgecolor="none", zorder=2,
            )
            ax.add_patch(rect)
            text_color = "#FFFFFF" if _luminance(color) < 0.52 else FG
            ax.text(
                cx + 0.5, ry + 0.5, f"{v:.2f}".lstrip("0") if v < 1 else "1.00",
                fontsize=8.6, color=text_color, ha="center", va="center",
                zorder=3, fontweight="500",
            )

    # --- row labels + identity dots
    for n in rows:
        ry = row_y[n] + 0.5
        parsed = parse_llm_player(n)
        if parsed:
            model, persona = parsed
            ax.scatter([-0.42], [ry], s=60, c=bases[model], marker="D",
                       edgecolors="white", linewidths=1.0, zorder=4, clip_on=False)
            label = f"{model}  ·  {persona}"
        else:
            ax.scatter([-0.42], [ry], s=48, c=CLASSIC, marker="o",
                       edgecolors="white", linewidths=1.0, zorder=4, clip_on=False)
            label = display_name_classic(n)
        ax.text(-0.62, ry, label, fontsize=9.3, color=FG, ha="right", va="center")

    # --- section dividers + labels
    ax.plot([-2.55, col_x[-1] + 1.05], [block_gap_y, block_gap_y],
            color=GRID, linewidth=1.0, zorder=1)
    for dy in model_dividers:
        ax.plot([-0.75, col_x[-1] + 1.05], [dy, dy],
                color=GRID, linewidth=0.8, linestyle=(0, (1, 0)), alpha=0.7, zorder=1)

    ax.text(-2.55, -1.55, "CLASSIC STRATEGIES", fontsize=8.5, fontweight="600",
            color=MUTED, ha="left", va="center")
    ax.text(-2.55, block_gap_y + 0.6, "LLM PLAYERS", fontsize=8.5, fontweight="600",
            color=MUTED, ha="left", va="center")
    ax.text(-2.55, block_gap_y + 0.92, "by model, cooperative -> selfish",
            fontsize=7.6, fontweight="400", color=MUTED, ha="left", va="center")

    # --- column headers (two-tier: trait group, then metric)
    dov_c = (col_x[0] + col_x[2] + 1) / 2
    hawk_c = (col_x[3] + col_x[4] + 1) / 2
    ax.text(dov_c, -1.55, "DOVISH", fontsize=8.2, fontweight="600",
            color=MUTED, ha="center", va="center")
    ax.text(hawk_c, -1.55, "HAWKISH", fontsize=8.2, fontweight="600",
            color=MUTED, ha="center", va="center")
    for j, (key, label) in enumerate(COLUMNS):
        ax.text(col_x[j] + 0.5, -0.9, label, fontsize=9.5, fontweight="600",
                color=FG, ha="center", va="center")

    ax.set_xlim(-2.75, col_x[-1] + 1.25)
    ax.set_ylim(total_y + 0.2, -2.05)

    add_title_block(
        fig, "Behavioral fingerprint",
        "Five dimensions, read like ink on paper — darker means more of the trait",
        left=0.055,
    )
    fig.text(
        0.055, 0.125,
        "Classics sorted by cooperation rate; LLM players grouped by model,\n"
        "ordered cooperative -> neutral -> payoff-only -> selfish.",
        fontsize=8.3, color=MUTED, ha="left", va="top", linespacing=1.5,
    )

    # --- sequential scale legend (bottom, its own clear band)
    _draw_sequential_legend(fig, left=0.055, bottom=0.028, width=0.30)

    fig.savefig(path)
    plt.close(fig)


def display_name_classic(name: str) -> str:
    return name


def _draw_sequential_legend(fig, *, left, bottom, width):
    overlay = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=10)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")

    n = 60
    for i in range(n):
        v = i / (n - 1)
        overlay.add_patch(FancyBboxPatch(
            (left + i * width / n, bottom), width / n * 1.05, 0.018,
            boxstyle="round,pad=0,rounding_size=0.0", linewidth=0,
            facecolor=blend_hex(HEAT_LOW, HEAT_HIGH, v),
            transform=fig.transFigure, clip_on=False,
        ))
    overlay.text(left, bottom + 0.032, "0.0  (never)", fontsize=8, color=MUTED,
                 ha="left", va="center", transform=fig.transFigure)
    overlay.text(left + width, bottom + 0.032, "1.0  (always)", fontsize=8, color=MUTED,
                 ha="right", va="center", transform=fig.transFigure)


# --- figure 2: leaderboard dumbbell ----------------------------------------

def draw_leaderboard_dumbbell(outcomes, path):
    wins = {n: o["wins"] for n, o in outcomes.items()}
    utility = {n: o["mean_score_per_turn"] for n, o in outcomes.items()}
    wins_rank = competition_ranks(wins)
    util_rank = competition_ranks(utility)

    names = sorted(outcomes.keys(), key=lambda n: (util_rank[n], n))
    n_rows = len(names)

    fig = plt.figure(figsize=(9.6, 7.9))
    ax = fig.add_axes([0.30, 0.30, 0.64, 0.56])
    ax.set_facecolor(BG)

    max_wins = max(wins.values()) or 1

    for i, name in enumerate(names):
        y = n_rows - 1 - i
        wr, ur = wins_rank[name], util_rank[name]
        is_llm = parse_llm_player(name) is not None
        marker = "D" if is_llm else "o"

        ax.plot([wr, ur], [y, y], color=SPINE, linewidth=1.6, zorder=1,
                solid_capstyle="round")

        w_size = 55 + (wins[name] / max_wins) * 260
        ax.scatter([wr], [y], s=w_size, c="#FFFFFF", marker=marker,
                   edgecolors=CLASSIC, linewidths=1.3, zorder=2)
        ax.scatter([ur], [y], s=140, c=FG, marker=marker,
                   edgecolors="white", linewidths=1.2, zorder=3)

        ax.text(ur + 0.42, y, f"{utility[name]:.2f}", fontsize=8.4,
                color=FG, ha="left", va="center", fontweight="600", zorder=3)
        if wins[name] >= 20:
            ax.text(wr, y - 0.34, f"{wins[name]} wins", fontsize=7.6,
                    color=MUTED, ha="center", va="top", zorder=3)

        label = _short_name(name) if is_llm else name
        color = FG
        ax.text(0.3, y, label, fontsize=9.2, color=color, ha="right", va="center")

    ax.set_xlim(0.5, n_rows + 1.6)
    ax.set_ylim(-1, n_rows)
    ax.set_xticks(range(1, n_rows + 1))
    ax.tick_params(axis="x", length=0, pad=8, labelsize=8.5, colors=MUTED)
    ax.set_yticks([])
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Tournament rank (1 = best, 15 = worst)", labelpad=10,
                  color=MUTED, fontsize=10)

    add_title_block(
        fig, "Winning isn't the same as scoring",
        "Pale = rank by head-to-head match wins.  Ink = rank by mean payoff per turn.",
        left=0.055,
    )
    fig.text(
        0.055, 0.155,
        "Sorted by mean payoff/turn — the metric that actually compounds across a\n"
        "long-run population. A long connecting line is a strategy whose win count\n"
        "lies to you about how well it actually did.",
        fontsize=8.3, color=MUTED, ha="left", va="top", linespacing=1.5,
    )

    _draw_dumbbell_legend(fig, left=0.055, bottom=0.032)

    fig.savefig(path)
    plt.close(fig)


def _draw_dumbbell_legend(fig, *, left, bottom):
    overlay = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=10)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")

    row_y = bottom + 0.026
    x = left
    overlay.scatter([x], [row_y], s=70, c="#FFFFFF", marker="o",
                     edgecolors=CLASSIC, linewidths=1.3,
                     transform=fig.transFigure, clip_on=False, zorder=3)
    overlay.text(x + 0.014, row_y, "rank by wins", fontsize=8.5, color=MUTED,
                 ha="left", va="center", transform=fig.transFigure)
    x += 0.115
    overlay.scatter([x], [row_y], s=120, c=FG, marker="o",
                     edgecolors="white", linewidths=1.2,
                     transform=fig.transFigure, clip_on=False, zorder=3)
    overlay.text(x + 0.014, row_y, "rank by payoff/turn", fontsize=8.5, color=MUTED,
                 ha="left", va="center", transform=fig.transFigure)
    x += 0.155
    overlay.scatter([x], [row_y], s=70, c=MUTED, marker="o",
                     edgecolors="white", linewidths=1.0,
                     transform=fig.transFigure, clip_on=False, zorder=3)
    overlay.text(x + 0.014, row_y, "classic strategy", fontsize=8.5, color=MUTED,
                 ha="left", va="center", transform=fig.transFigure)
    x += 0.135
    overlay.scatter([x], [row_y], s=70, c=MUTED, marker="D",
                     edgecolors="white", linewidths=1.0,
                     transform=fig.transFigure, clip_on=False, zorder=3)
    overlay.text(x + 0.014, row_y, "LLM player", fontsize=8.5, color=MUTED,
                 ha="left", va="center", transform=fig.transFigure)

    panel = FancyBboxPatch(
        (left - 0.014, bottom - 0.004), 0.62, 0.058,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        transform=fig.transFigure, facecolor="#FFFFFF", edgecolor=GRID,
        linewidth=1.0, zorder=1, clip_on=False,
    )
    overlay.add_patch(panel)


# --- figure 3: persona small multiples --------------------------------------

PERSONA_LABELS = {
    "cooperative": "cooperative",
    "neutral": "neutral",
    "payoff_only": "payoff-only",
    "selfish": "selfish",
}


def draw_persona_slope(fingerprints, fp_order, path):
    llms = [n for n in fp_order if parse_llm_player(n) is not None]
    bases = model_base_colors(llms)
    models = []
    for n in llms:
        m = parse_llm_player(n)[0]
        if m not in models:
            models.append(m)

    fig, axes = plt.subplots(1, len(models), figsize=(9.6, 4.9), sharey=True)
    fig.subplots_adjust(top=0.77, bottom=0.24, left=0.11, right=0.97, wspace=0.12)

    xs = list(range(len(PERSONA_ORDER)))
    for ax, model in zip(axes, models):
        color = bases[model]
        ys = [fingerprints[llm_player_name(model, p)]["cooperation_rate"] for p in PERSONA_ORDER]

        ax.plot(xs, ys, color=color, linewidth=2.2, zorder=2, solid_capstyle="round")
        ax.scatter(xs, ys, s=150, c=color, marker="D", edgecolors="white",
                   linewidths=1.4, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.06, f"{y:.2f}", fontsize=9.5, fontweight="600",
                    color=FG, ha="center", va="bottom", zorder=4)

        drop = ys[0] - ys[-1]
        ax.text(0.5, 1.16, model, transform=ax.transAxes, fontsize=13,
                fontweight="600", color=FG, ha="center", va="bottom")
        ax.text(0.5, 1.05, f"cooperative -> selfish:  −{drop:.2f}",
                transform=ax.transAxes, fontsize=9, color=MUTED, ha="center", va="bottom")

        ax.set_xlim(-0.5, len(PERSONA_ORDER) - 0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([PERSONA_LABELS[p] for p in PERSONA_ORDER], fontsize=9.5)
        ax.tick_params(axis="x", length=0, pad=8, colors=MUTED)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylim(-0.06, 1.14)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].tick_params(axis="y", length=0, pad=6, colors=MUTED)
    axes[0].set_ylabel("Cooperation rate", labelpad=10, color=MUTED, fontsize=10)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0, labelleft=False)

    add_title_block(
        fig, "Persona moves the needle, model barely does",
        "The same four instructions, on two different models, produce nearly the same swing.",
    )
    fig.text(
        0.11, 0.075,
        "Four system prompts, same models, temperature 0. Both curves fall by roughly\n"
        "the same amount across the same personas — the instruction predicts cooperation\n"
        "far better than which model received it.",
        fontsize=8.3, color=MUTED, ha="left", va="top", linespacing=1.5,
    )

    fig.savefig(path)
    plt.close(fig)


# --- figure 4: sample match transcripts ---------------------------------------

SAMPLE_MATCHES = [
    ("LLM:claude-haiku-4-5[neutral]", "Defector",
     "Cooperates once, then defects for the rest of the match."),
    ("LLM:claude-haiku-4-5[cooperative]", "Defector",
     "Forgives once more before it adapts — one extra costly round."),
    ("LLM:gpt-4o-mini[payoff_only]", "Tit For Tat",
     "Can't find mutual cooperation even against a purely reactive opponent."),
]


def draw_sample_transcripts(interactions_csv, path):
    loaded = [
        (me, opp, caption, load_sample_match(interactions_csv, me, opp))
        for me, opp, caption in SAMPLE_MATCHES
    ]
    n_turns = max(m["turns"] for _, _, _, m in loaded)

    fig = plt.figure(figsize=(9.9, 6.9))
    ax = fig.add_axes([0.235, 0.135, 0.62, 0.68])
    ax.set_facecolor(BG)
    ax.axis("off")

    block_h = 2.0
    gap = 1.15
    cell = 0.86

    # turn ticks sit above the first block, like column headers elsewhere
    for t in range(0, n_turns, 5):
        ax.text(t + 0.5, -1.35, str(t + 1), fontsize=7.4, color=MUTED,
                ha="center", va="bottom")
    ax.text(n_turns / 2, -1.75, "Turn", fontsize=8.5, color=MUTED,
            ha="center", va="bottom")

    y = 0.0
    for me, opp, caption, m in loaded:
        for row_i, seq in enumerate((m["mine"], m["theirs"])):
            ry = y + row_i * 1.0
            for t, ch in enumerate(seq):
                color = HEAT_HIGH if ch.upper() == "D" else "#F1F2F4"
                rect = FancyBboxPatch(
                    (t + (1 - cell) / 2, ry + (1 - cell) / 2), cell, cell,
                    boxstyle="round,pad=0,rounding_size=0.05",
                    facecolor=color, edgecolor="none", zorder=2,
                )
                ax.add_patch(rect)
            label = _short_name(me) if row_i == 0 else opp
            ax.text(-0.4, ry + 0.5, label, fontsize=8.4, color=FG, ha="right", va="center")

        ax.text(-0.4, y + 2.2, caption, fontsize=8.3, color=MUTED, ha="left", va="top")
        score_text = f"{m['total_score']:.0f} pts  ·  {m['score_per_turn']:.2f}/turn"
        ax.text(n_turns + 0.7, y + 0.5, score_text, fontsize=8.4, color=FG,
                ha="left", va="center", fontweight="600")

        y += block_h + gap

    total_h = y - gap

    ax.set_xlim(-4.7, n_turns + 4.2)
    ax.set_ylim(total_h + 0.4, -2.15)

    # legend: cooperate / defect swatches
    overlay = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=10)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    ly = 0.045
    overlay.add_patch(FancyBboxPatch((0.235, ly), 0.016, 0.016,
                       boxstyle="round,pad=0,rounding_size=0.003",
                       facecolor="#F1F2F4", edgecolor="none",
                       transform=fig.transFigure, clip_on=False))
    overlay.text(0.256, ly + 0.008, "cooperate", fontsize=8.5, color=MUTED,
                 ha="left", va="center", transform=fig.transFigure)
    overlay.add_patch(FancyBboxPatch((0.345, ly), 0.016, 0.016,
                       boxstyle="round,pad=0,rounding_size=0.003",
                       facecolor=HEAT_HIGH, edgecolor="none",
                       transform=fig.transFigure, clip_on=False))
    overlay.text(0.366, ly + 0.008, "defect", fontsize=8.5, color=MUTED,
                 ha="left", va="center", transform=fig.transFigure)
    overlay.text(0.46, ly + 0.008, "top row = the LLM · bottom row = opponent",
                 fontsize=8.5, color=MUTED, ha="left", va="center",
                 transform=fig.transFigure)

    add_title_block(
        fig, "Three matches, move by move",
        "Real transcripts — every square is one round's actual choice, temperature 0.",
    )

    fig.savefig(path)
    plt.close(fig)


# --- figure 5: cooperation matrix heatmap -------------------------------------

MODEL_TAG = {"gpt-4o-mini": "gpt", "claude-haiku-4-5": "cld"}
PERSONA_TAG = {"cooperative": "c", "neutral": "n", "payoff_only": "p", "selfish": "s"}
CLASSIC_TAG = {
    "Cooperator": "Coop", "Win-Stay Lose-Shift": "WSLS", "GTFT: 0.33": "GTFT",
    "Tit For Tat": "TFT", "Grudger": "Grdg", "Random: 0.5": "Rand", "Defector": "Defe",
}


def draw_cooperation_heatmap(matrix, fingerprints, fp_order, path):
    classics, llms = ordered_rows(fp_order, fingerprints)
    order = classics + llms
    bases = model_base_colors(llms)
    n = len(order)

    def tag(name):
        parsed = parse_llm_player(name)
        if parsed:
            model, persona = parsed
            return f"{MODEL_TAG[model]}-{PERSONA_TAG[persona]}"
        return CLASSIC_TAG.get(name, name)

    fig = plt.figure(figsize=(9.9, 9.5))
    ax = fig.add_axes([0.225, 0.115, 0.60, 0.68])
    ax.set_facecolor(BG)
    ax.axis("off")

    cell = 0.88
    for i, row_name in enumerate(order):
        for j, col_name in enumerate(order):
            v = max(0.0, min(1.0, matrix[row_name][col_name]))
            color = blend_hex(HEAT_LOW, HEAT_HIGH, v)
            rect = FancyBboxPatch(
                (j + (1 - cell) / 2, i + (1 - cell) / 2), cell, cell,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=color, edgecolor="none", zorder=2,
            )
            ax.add_patch(rect)

    for i, name in enumerate(order):
        parsed = parse_llm_player(name)
        ry = i + 0.5
        if parsed:
            model, persona = parsed
            ax.scatter([-0.4], [ry], s=40, c=bases[model], marker="D",
                       edgecolors="white", linewidths=0.9, zorder=4, clip_on=False)
            label = f"{model} · {persona}"
        else:
            ax.scatter([-0.4], [ry], s=34, c=CLASSIC, marker="o",
                       edgecolors="white", linewidths=0.9, zorder=4, clip_on=False)
            label = name
        ax.text(-0.6, ry, label, fontsize=7.6, color=FG, ha="right", va="center")

    for j, name in enumerate(order):
        cx = j + 0.5
        ax.text(cx, -0.55, tag(name), fontsize=6.6, color=FG, ha="right", va="bottom",
                rotation=80, rotation_mode="anchor")

    n_classic = len(classics)
    models = []
    for name in llms:
        m = parse_llm_player(name)[0]
        if m not in models:
            models.append(m)
    sub = n_classic + sum(1 for name in llms if parse_llm_player(name)[0] == models[0])

    ax.plot([n_classic, n_classic], [-3.7, n], color=GRID, linewidth=1.1, zorder=1)
    ax.plot([-3.7, n], [n_classic, n_classic], color=GRID, linewidth=1.1, zorder=1)
    ax.plot([sub, sub], [n_classic, n], color=GRID, linewidth=0.7, alpha=0.6, zorder=1)
    ax.plot([n_classic, n], [sub, sub], color=GRID, linewidth=0.7, alpha=0.6, zorder=1)

    ax.set_xlim(-5.0, n + 0.4)
    ax.set_ylim(n + 0.4, -4.2)

    add_title_block(
        fig, "Who cooperates with whom",
        "Row = that player's cooperation rate when facing the column player. Not symmetric.",
    )
    fig.text(
        0.225, 0.112,
        "Same order as the fingerprint chart: classics by cooperation rate, then LLM\n"
        "players grouped by model. Exact values: results/data/cooperation_matrix.csv.",
        fontsize=8.3, color=MUTED, ha="left", va="top", linespacing=1.5,
    )
    _draw_sequential_legend(fig, left=0.225, bottom=0.026, width=0.30)

    fig.savefig(path)
    plt.close(fig)


def main():
    apply_theme()
    os.makedirs(FIGURES, exist_ok=True)
    fingerprints, fp_order = load_fingerprints()
    outcomes = load_outcomes()
    cooperation_matrix = load_cooperation_matrix()
    interactions_csv = os.path.join(DATA, "interactions_raw.csv")

    figures = [
        ("fingerprint_heatmap.png",
         lambda p: draw_fingerprint_heatmap(fingerprints, fp_order, p)),
        ("leaderboard_dumbbell.png",
         lambda p: draw_leaderboard_dumbbell(outcomes, p)),
        ("persona_slope.png",
         lambda p: draw_persona_slope(fingerprints, fp_order, p)),
        ("sample_transcripts.png",
         lambda p: draw_sample_transcripts(interactions_csv, p)),
        ("cooperation_heatmap.png",
         lambda p: draw_cooperation_heatmap(cooperation_matrix, fingerprints, fp_order, p)),
    ]
    for filename, draw in figures:
        draw(os.path.join(FIGURES, filename))
        print(f"Wrote results/figures/{filename}")


if __name__ == "__main__":
    main()
