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


def hero_figure(fps, llm_names, outpath):
    """Strategy space: x = cooperation_rate, y = forgiveness, marker size ~
    retaliation. Classics as labeled dots, LLMs as stars."""
    fig, ax = plt.subplots(figsize=(11, 9))
    texts = []
    for name, fp in fps.items():
        x = _plot_value(fp["cooperation_rate"], 0.5)
        # Undefined forgiveness (no opponent defections to respond to) sits with
        # the maximally cooperative cluster rather than breaking label placement.
        y = _plot_value(fp["forgiveness"], 1.0)
        is_llm = name.startswith("LLM:") or name in llm_names
        ax.scatter(
            x, y,
            s=280 if is_llm else 130,
            marker="*" if is_llm else "o",
            edgecolor="black", linewidth=0.8,
            zorder=3 if is_llm else 2,
        )
        texts.append(ax.text(
            x, y, _display_name(name),
            fontsize=9,
            fontweight="bold" if is_llm else "normal",
        ))
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6, shrinkA=4),
        expand=(1.25, 1.5),
        force_text=(0.8, 1.0),
        force_points=(0.3, 0.5),
        only_move={"text": "xy", "points": "y", "objects": "xy"},
    )
    ax.set_xlabel("Cooperation rate  (nice  \u2192)")
    ax.set_ylabel("Forgiveness  (rebuilds cooperation  \u2192)")
    ax.set_title("Behavioral fingerprint: LLMs (\u2605) vs classic strategies (\u25cf)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-0.08, 1.12)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
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
