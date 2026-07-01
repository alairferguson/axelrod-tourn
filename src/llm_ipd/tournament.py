"""Run the tournament: classic reference strategies plus LLM players.

Produces:
  - results/data/ranked_scores.csv         (final ranking + normalised scores)
  - results/data/cooperation_matrix.csv    (who cooperates against whom)
  - results/data/interactions.pkl          (raw move histories for fingerprinting)

The LLM players are stateless across matches and see moves-only prompts (enforced
by LLMPlayer), so they sit in the tournament on the same footing as the classics.
"""

from __future__ import annotations

import argparse
import os
import pickle

import axelrod as axl

from .cache import ResponseCache
from .llm_player import LLMPlayer, llm_player_name
from .prompts import PERSONA_PROMPTS
from .roster import classic_roster


def build_players(models, personas=("neutral",), temperature=0.0, cache=None):
    """Build classics plus one distinct player per model×persona pairing."""
    players = classic_roster()
    for model in models:
        for persona in personas:
            players.append(
                LLMPlayer(
                    model=model,
                    system_prompt=PERSONA_PROMPTS[persona],
                    temperature=temperature,
                    cache=cache,
                    name=llm_player_name(model, persona),
                    persona=persona,
                )
            )
    return players


def run(models, turns=30, repetitions=5, personas=("neutral",),
        temperature=0.0, seed=0, outdir="results/data"):
    os.makedirs(outdir, exist_ok=True)
    cache = ResponseCache(os.path.join(outdir, "llm_cache.json"))
    players = build_players(models, personas, temperature, cache)

    print(f"Players ({len(players)}): {[p.name for p in players]}")
    print(f"turns={turns} repetitions={repetitions} personas={list(personas)} "
          f"temperature={temperature}")

    tournament = axl.Tournament(
        players, turns=turns, repetitions=repetitions, seed=seed
    )
    # Serial play: LLMPlayer's cache is not multiprocess-safe.
    # filename= records every match's move sequence to CSV (the source we
    # fingerprint from).
    interactions_csv = os.path.join(outdir, "interactions_raw.csv")
    results = tournament.play(
        processes=None, progress_bar=True, filename=interactions_csv
    )
    cache.save()
    print(f"Cache now holds {len(cache)} responses.")

    # --- Save ranked scores ---
    import csv
    ranked = list(zip(results.ranked_names, results.normalised_scores))
    with open(os.path.join(outdir, "ranked_scores.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "player", "mean_normalised_score"])
        for i, (name, scores) in enumerate(ranked, start=1):
            mean = sum(scores) / len(scores) if scores else float("nan")
            w.writerow([i, name, f"{mean:.4f}"])

    # --- Save cooperation matrix ---
    coop = results.normalised_cooperation
    names = results.players
    with open(os.path.join(outdir, "cooperation_matrix.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + names)
        for i, row in enumerate(coop):
            w.writerow([names[i]] + [f"{v:.3f}" for v in row])

    # --- Build per-player match histories for fingerprinting ---
    # Parse the interactions CSV: each row has Player name, Opponent name, and an
    # "Actions" string from the player's perspective (e.g. "CDDC...").
    from collections import defaultdict
    from axelrod.action import Action as _A

    def _to_pairs(actions_str):
        # "Actions" encodes this player's moves; we need (mine, theirs) pairs.
        # The CSV stores one row per player-perspective, so the opponent's row
        # gives theirs. We reconstruct pairs by matching the two perspectives.
        return [c for c in actions_str]

    by_player = defaultdict(list)  # player name -> list of own move strings
    rows = []
    with open(interactions_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
            by_player[row["Player name"]].append(row["Actions"])

    with open(os.path.join(outdir, "interactions.pkl"), "wb") as fh:
        pickle.dump(
            {"players": names, "by_player_moves": dict(by_player), "rows": rows},
            fh,
        )

    print(f"Wrote results to {outdir}/")
    return results


def main():
    ap = argparse.ArgumentParser(description="Run the LLM-IPD tournament.")
    ap.add_argument("--models", nargs="*", default=[],
                    help="litellm model strings, e.g. gpt-4o-mini claude-haiku-4-5")
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument(
        "--personas", nargs="*", default=None, choices=list(PERSONA_PROMPTS),
        help="One player per model×persona (default: neutral). "
             "E.g. --personas neutral cooperative selfish",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    personas = tuple(args.personas if args.personas else ("neutral",))
    run(args.models, args.turns, args.repetitions, personas,
        args.temperature, args.seed)


if __name__ == "__main__":
    main()
