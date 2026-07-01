"""Analysis + figures.

Reads the tournament interactions CSV, computes behavioral fingerprints and
outcome stats, finds each LLM's nearest classic strategy via probe profiles, and
draws publication figures.

Run after tournament.py:
    python -m llm_ipd.analyze --models gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import math
import os

from dotenv import load_dotenv

load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

from .fingerprint import fingerprint, probe_profile, nearest_strategy
from .plot_style import (
    BG, CLASSIC, FG, GRID, LLM_ACCENTS, MUTED, SPINE,
    add_title_block, apply_theme, build_llm_colors,
    draw_fingerprint_legend, marker_size, style_axes,
)
from .io_utils import load_player_matches, load_player_outcomes
from .roster import classic_factories
from .cache import ResponseCache
from .llm_player import LLMPlayer, display_llm_name, llm_player_name
from .prompts import PERSONA_PROMPTS


def compute_fingerprints(interactions_csv):
    matches_by_player = load_player_matches(interactions_csv)
    return {name: fingerprint(matches) for name, matches in matches_by_player.items()}


def build_classic_profiles():
    return {name: probe_profile(factory)
            for name, factory in classic_factories().items()}


def llm_factory(model, persona, temperature, cache):
    sp = PERSONA_PROMPTS[persona]
    return lambda: LLMPlayer(model=model, system_prompt=sp,
                             temperature=temperature, cache=cache,
                             persona=persona,
                             name=llm_player_name(model, persona))


def _display_name(name: str) -> str:
    return display_llm_name(name)


def _plot_value(value: float, nan_fallback: float) -> float:
    return value if math.isfinite(value) else nan_fallback


def _label_offset(x: float, y: float, idx: int) -> tuple[float, float]:
    """Seed labels slightly outward from plot center, with light angular jitter."""
    angles = (0.85, 2.1, 4.0, 5.2, 0.4, 3.6, 1.7, 4.8, 2.8)
    dist = 0.048
    cx, cy = 0.5, 0.5
    dx, dy = x - cx, y - cy
    norm = math.hypot(dx, dy) or 1.0
    radial_x = x + dist * dx / norm
    radial_y = y + dist * dy / norm
    angle = angles[idx % len(angles)]
    angled_x = x + dist * math.cos(angle)
    angled_y = y + dist * math.sin(angle)
    return 0.6 * radial_x + 0.4 * angled_x, 0.6 * radial_y + 0.4 * angled_y


def hero_figure(fps, llm_names, outpath):
    """Strategy space: x = cooperation_rate, y = forgiveness, marker size ~
    retaliation. Classics as labeled dots, LLMs as stars."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(BG)

    llm_names_in_fps = [
        name for name in fps
        if name.startswith("LLM:") or name in llm_names
    ]
    llm_colors = build_llm_colors(llm_names_in_fps)

    texts = []
    xs, ys = [], []

    # Plot classics first so LLM markers sit on top.
    for idx, (name, fp) in enumerate(
        sorted(fps.items(), key=lambda kv: kv[0].startswith("LLM:"))
    ):
        x = _plot_value(fp["cooperation_rate"], 0.5)
        # Undefined forgiveness (no opponent defections to respond to) sits with
        # the maximally cooperative cluster rather than breaking label placement.
        y = _plot_value(fp["forgiveness"], 1.0)
        is_llm = name.startswith("LLM:") or name in llm_names
        retaliation = _plot_value(fp["retaliation"], 0.5)
        size = marker_size(retaliation, llm=is_llm)
        xs.append(x)
        ys.append(y)

        if is_llm:
            color = llm_colors.get(name, LLM_ACCENTS[0])
            ax.scatter(
                x, y, s=size * 2.2, c=color, alpha=0.12,
                marker="o", linewidths=0, zorder=2,
            )
            ax.scatter(
                x, y, s=size, c=color, marker="D",
                edgecolors="white", linewidths=1.6, zorder=4,
            )
            label_color = color
            label_weight = "600"
        else:
            ax.scatter(
                x, y, s=size, c=CLASSIC, marker="o", alpha=0.85,
                edgecolors="white", linewidths=1.2, zorder=3,
            )
            label_color = FG
            label_weight = "400"

        lx, ly = _label_offset(x, y, idx)
        if is_llm:
            lx += 0.022
            ly += 0.008
        texts.append(ax.text(
            lx, ly, _display_name(name),
            fontsize=9.5 if is_llm else 9,
            fontweight=label_weight,
            color=label_color if is_llm else MUTED,
            ha="left" if is_llm else "center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=BG,
                edgecolor="none",
                alpha=0.92,
            ),
            zorder=5,
        ))

    adjust_text(
        texts,
        x=xs,
        y=ys,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.7, shrinkA=5, shrinkB=2),
        expand=(1.15, 1.35),
        expand_points=(1.6, 1.6),
        expand_text=(1.15, 1.25),
        force_points=(1.1, 1.3),
        force_text=(0.5, 0.7),
        only_move={"text": "xy", "points": "y", "objects": "xy"},
        ensure_inside_axes=True,
        lim=400,
    )

    style_axes(
        ax,
        xlabel="Cooperation rate",
        ylabel="Forgiveness",
    )
    add_title_block(
        fig,
        "Behavioral fingerprint",
        "LLM players in iterated prisoner's dilemma strategy space",
    )
    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(-0.06, 1.08)

    draw_fingerprint_legend(fig, llm_colors=llm_colors)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.17)
    fig.savefig(outpath, facecolor=BG)
    print(f"Wrote {outpath}")


def _player_sort_key(name: str, outcomes) -> tuple:
    """Classics first (stable), then LLMs sorted by utility descending."""
    outcome = outcomes[name]
    is_llm = name.startswith("LLM:")
    return (is_llm, -outcome.mean_score_per_turn, name.lower())


def _leaderboard_sort_key(name: str, outcomes) -> tuple:
    """Rank by total tournament score, best first."""
    return (-outcomes[name].total_score, name.lower())


def _bar_label(value: float, *, integer: bool = False) -> str:
    if integer:
        return str(int(round(value)))
    return f"{value:.2f}"


def outcomes_figure(outcomes, llm_names, outpath):
    """Dual-panel horizontal bars: tournament wins and mean score per turn."""
    apply_theme()

    names = sorted(outcomes.keys(), key=lambda n: _player_sort_key(n, outcomes))
    llm_names_in_chart = [n for n in names if n.startswith("LLM:") or n in llm_names]
    llm_colors = build_llm_colors(llm_names_in_chart)

    wins = [outcomes[n].wins for n in names]
    utils = [outcomes[n].mean_score_per_turn for n in names]
    colors = [
        llm_colors.get(n, CLASSIC) if (n.startswith("LLM:") or n in llm_names)
        else CLASSIC
        for n in names
    ]
    labels = [_display_name(n) for n in names]

    n = len(names)
    y = list(range(n))
    bar_h = 0.62
    fig_h = max(7.5, 0.44 * n + 2.4)

    fig = plt.figure(figsize=(13.5, fig_h))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.08], wspace=0.06)
    ax_wins = fig.add_subplot(gs[0, 0])
    ax_util = fig.add_subplot(gs[0, 1], sharey=ax_wins)

    for ax in (ax_wins, ax_util):
        ax.set_facecolor(BG)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)
        ax.spines["bottom"].set_linewidth(0.8)

    for i in range(n):
        if i % 2 == 0:
            for ax in (ax_wins, ax_util):
                ax.axhspan(
                    i - 0.5, i + 0.5, color="#F0F3F6", alpha=0.9, zorder=0,
                )

    llm_start = next(
        (i for i, name in enumerate(names) if name.startswith("LLM:")), n,
    )
    if 0 < llm_start < n:
        for ax in (ax_wins, ax_util):
            ax.axhline(llm_start - 0.5, color=GRID, linewidth=1.4, zorder=1)
        ax_wins.text(
            -0.02, llm_start - 0.5, "LLM players",
            transform=ax_wins.get_yaxis_transform(),
            fontsize=8, color=MUTED, ha="right", va="bottom",
            fontweight="500",
        )

    ax_wins.barh(
        y, wins, height=bar_h, color=colors, edgecolor="white",
        linewidth=1.2, zorder=3,
    )
    ax_util.barh(
        y, utils, height=bar_h, color=colors, edgecolor="white",
        linewidth=1.2, alpha=0.92, zorder=3,
    )
    ax_util.axvline(
        3.0, color=GRID, linewidth=1.0, linestyle=(0, (4, 4)), zorder=2,
    )
    ax_util.text(
        3.0, 1.012, "mutual coop.",
        transform=ax_util.get_xaxis_transform(),
        fontsize=7.5, color=MUTED, ha="center", va="bottom", style="italic",
    )

    max_wins = max(wins) if wins else 1
    max_util = max(utils) if utils else 3.0
    ax_wins.set_xlim(0, max(max_wins * 1.18, 1))
    ax_util.set_xlim(0, min(5.05, max(max_util * 1.14, 3.35)))

    for yi, val in zip(y, wins):
        if val > 0:
            ax_wins.text(
                val + max_wins * 0.02, yi, _bar_label(val, integer=True),
                va="center", ha="left", fontsize=9, color=FG, fontweight="500",
                zorder=4,
            )
    for yi, val, color in zip(y, utils, colors):
        ax_util.text(
            val + 0.04, yi, _bar_label(val),
            va="center", ha="left", fontsize=9,
            color=color if color != CLASSIC else FG,
            fontweight="600" if color != CLASSIC else "500",
            zorder=4,
        )

    ax_wins.set_yticks(y)
    ax_wins.set_yticklabels(labels)
    for tick, name in zip(ax_wins.get_yticklabels(), names):
        is_llm = name.startswith("LLM:") or name in llm_names
        tick.set_color(llm_colors.get(name, MUTED) if is_llm else MUTED)
        tick.set_fontweight("600" if is_llm else "400")
        tick.set_fontsize(9.5 if is_llm else 9)
    plt.setp(ax_util.get_yticklabels(), visible=False)

    ax_wins.invert_yaxis()
    ax_wins.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    ax_wins.grid(True, axis="x", alpha=0.45, linewidth=0.8)
    ax_util.grid(True, axis="x", alpha=0.45, linewidth=0.8)

    ax_wins.set_xlabel("Match wins", labelpad=10, fontweight="500")
    ax_util.set_xlabel("Mean score per turn", labelpad=10, fontweight="500")

    ax_wins.text(
        0.0, 1.015, "Head-to-head wins",
        transform=ax_wins.transAxes, fontsize=11, fontweight="600", color=FG,
        ha="left", va="bottom",
    )
    ax_util.text(
        0.0, 1.015, "Per-turn utility",
        transform=ax_util.transAxes, fontsize=11, fontweight="600", color=FG,
        ha="left", va="bottom",
    )

    add_title_block(
        fig,
        "Tournament outcomes",
        "Wins and average payoff per turn across all head-to-head matches",
    )

    left_margin = min(0.40, max(0.24, 0.0075 * max(len(l) for l in labels) + 0.12))
    fig.subplots_adjust(
        left=left_margin, right=0.97, top=0.86, bottom=0.08, wspace=0.10,
    )
    fig.savefig(outpath, facecolor=BG)
    print(f"Wrote {outpath}")


def utility_leaderboard_figure(outcomes, llm_names, outpath):
    """Single-panel horizontal bars ranked by total tournament utility."""
    apply_theme()

    names = sorted(outcomes.keys(), key=lambda n: _leaderboard_sort_key(n, outcomes))
    llm_names_in_chart = [n for n in names if n.startswith("LLM:") or n in llm_names]
    llm_colors = build_llm_colors(llm_names_in_chart)

    totals = [outcomes[n].total_score for n in names]
    colors = [
        llm_colors.get(n, CLASSIC) if (n.startswith("LLM:") or n in llm_names)
        else CLASSIC
        for n in names
    ]
    labels = [
        f"{rank} · {_display_name(n)}" for rank, n in enumerate(names, start=1)
    ]

    n = len(names)
    y = list(range(n))
    bar_h = 0.62
    fig_h = max(6.5, 0.44 * n + 2.0)
    fig_w = max(9.5, 0.045 * max(len(l) for l in labels) + 7.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["bottom"].set_linewidth(0.8)

    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F0F3F6", alpha=0.9, zorder=0)

    ax.barh(
        y, totals, height=bar_h, color=colors, edgecolor="white",
        linewidth=1.2, zorder=3,
    )

    max_total = max(totals) if totals else 1
    ax.set_xlim(0, max_total * 1.16)

    for yi, val, color in zip(y, totals, colors):
        is_llm = color != CLASSIC
        ax.text(
            val + max_total * 0.015, yi, _bar_label(val, integer=True),
            va="center", ha="left", fontsize=9.5,
            color=color if is_llm else FG,
            fontweight="600" if is_llm else "500",
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for tick, name in zip(ax.get_yticklabels(), names):
        is_llm = name.startswith("LLM:") or name in llm_names
        tick.set_color(llm_colors.get(name, MUTED) if is_llm else MUTED)
        tick.set_fontweight("600" if is_llm else "400")
        tick.set_fontsize(9.5 if is_llm else 9)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    ax.grid(True, axis="x", alpha=0.45, linewidth=0.8)
    ax.set_xlabel("Total score", labelpad=10, fontweight="500")

    add_title_block(
        fig,
        "Tournament leaderboard",
        "Total utility accumulated across all head-to-head matches",
    )

    left_margin = min(0.42, max(0.26, 0.0075 * max(len(l) for l in labels) + 0.14))
    fig.subplots_adjust(left=left_margin, right=0.97, top=0.88, bottom=0.08)
    fig.savefig(outpath, facecolor=BG)
    print(f"Wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument(
        "--personas", nargs="*", default=None, choices=list(PERSONA_PROMPTS),
        help="LLM×persona pairs for probe nearest-strategy (default: neutral)",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--datadir", default="results/data")
    ap.add_argument("--figdir", default="results/figures")
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    interactions_csv = os.path.join(args.datadir, "interactions_raw.csv")

    fps = compute_fingerprints(interactions_csv)
    outcomes = load_player_outcomes(interactions_csv)
    llm_names = {n for n in outcomes if n.startswith("LLM:")}

    print("\n=== Behavioral fingerprints ===")
    for name, fp in fps.items():
        print(f"{name:28s} " + "  ".join(f"{k}={v:.2f}" for k, v in fp.items()))

    # Save fingerprints
    with open(os.path.join(args.datadir, "fingerprints.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        keys = ["cooperation_rate", "niceness", "retaliation",
                "forgiveness", "provocability"]
        w.writerow(["player"] + keys)
        for name, fp in fps.items():
            w.writerow([name] + [f"{fp[k]:.4f}" for k in keys])

    with open(os.path.join(args.datadir, "outcomes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["player", "wins", "mean_score_per_turn", "total_score", "matches"])
        for name in sorted(outcomes.keys(), key=lambda n: _player_sort_key(n, outcomes)):
            o = outcomes[name]
            w.writerow([
                name, o.wins, f"{o.mean_score_per_turn:.4f}",
                f"{o.total_score:.1f}", o.matches,
            ])

    print("\n=== Tournament outcomes ===")
    for name in sorted(outcomes.keys(), key=lambda n: _leaderboard_sort_key(n, outcomes)):
        o = outcomes[name]
        print(
            f"{_display_name(name):32s}  wins={o.wins:3d}  "
            f"total={o.total_score:6.0f}  util/turn={o.mean_score_per_turn:.3f}  "
            f"matches={o.matches}"
        )

    # Nearest-strategy for each LLM (probe method)
    personas = tuple(args.personas if args.personas else ("neutral",))
    if args.models:
        cache = ResponseCache(os.path.join(args.datadir, "llm_cache.json"))
        classic_profiles = build_classic_profiles()
        print("\n=== Nearest classic strategy (probe method) ===")
        for model in args.models:
            for persona in personas:
                label = llm_player_name(model, persona)
                llm_names.add(label)
                prof = probe_profile(
                    llm_factory(model, persona, args.temperature, cache)
                )
                ranking = nearest_strategy(prof, classic_profiles)
                cache.save()
                top = ranking[0]
                print(f"{label}: closest to '{top[0]}' (distance {top[1]:.2f})")
                print("   full ranking: " +
                      ", ".join(f"{n}={d:.2f}" for n, d in ranking))

    hero_figure(fps, llm_names,
                os.path.join(args.figdir, "fingerprint_space.png"))
    outcomes_figure(outcomes, llm_names,
                    os.path.join(args.figdir, "tournament_outcomes.png"))
    utility_leaderboard_figure(outcomes, llm_names,
                               os.path.join(args.figdir, "utility_leaderboard.png"))


if __name__ == "__main__":
    main()
