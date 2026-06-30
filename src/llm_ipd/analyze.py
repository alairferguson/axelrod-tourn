"""Analysis + figures.

Reads the tournament interactions CSV, computes the behavioral fingerprint for
every player, finds each LLM's nearest classic strategy via probe profiles, and
draws the hero figure (strategy space with every player as a point).

Run after tournament.py:
    python -m llm_ipd.analyze --models gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

from .fingerprint import fingerprint, probe_profile, nearest_strategy
from .plot_style import (
    BG, CLASSIC, FG, GRID, LLM_ACCENTS, MUTED,
    add_title_block, apply_theme, draw_fingerprint_legend, marker_size, style_axes,
)
from .io_utils import load_player_matches
from .roster import classic_factories
from .cache import ResponseCache
from .llm_player import LLMPlayer
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
                             name=f"LLM:{model.split('/')[-1]}")


def _display_name(name: str) -> str:
    if name.startswith("LLM:"):
        return name[4:]
    return name


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

    llm_idx = 0
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
            color = LLM_ACCENTS[llm_idx % len(LLM_ACCENTS)]
            llm_idx += 1
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
    ax.text(0.98, 0.02, "more cooperative", fontsize=8.5, color=MUTED,
            ha="right", va="bottom", style="italic", zorder=1)
    ax.text(0.02, 0.98, "more forgiving", fontsize=8.5, color=MUTED,
            ha="left", va="top", style="italic", zorder=1)

    draw_fingerprint_legend(fig)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.17)
    fig.savefig(outpath, facecolor=BG)
    print(f"Wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--persona", default="neutral", choices=list(PERSONA_PROMPTS))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--datadir", default="results/data")
    ap.add_argument("--figdir", default="results/figures")
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    interactions_csv = os.path.join(args.datadir, "interactions_raw.csv")

    fps = compute_fingerprints(interactions_csv)
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

    # Nearest-strategy for each LLM (probe method)
    llm_names = set()
    if args.models:
        cache = ResponseCache(os.path.join(args.datadir, "llm_cache.json"))
        classic_profiles = build_classic_profiles()
        print("\n=== Nearest classic strategy (probe method) ===")
        for model in args.models:
            label = f"LLM:{model.split('/')[-1]}"
            llm_names.add(label)
            prof = probe_profile(
                llm_factory(model, args.persona, args.temperature, cache)
            )
            ranking = nearest_strategy(prof, classic_profiles)
            cache.save()
            top = ranking[0]
            print(f"{label}: closest to '{top[0]}' (distance {top[1]:.2f})")
            print("   full ranking: " +
                  ", ".join(f"{n}={d:.2f}" for n, d in ranking))

    hero_figure(fps, llm_names,
                os.path.join(args.figdir, "fingerprint_space.png"))


if __name__ == "__main__":
    main()
