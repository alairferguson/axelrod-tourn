"""Reconstruct per-player match histories as (mine, theirs) Action pairs from the
tournament interactions CSV. Both player perspectives share an Interaction index,
so we join them to recover full move pairs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from axelrod.action import Action

C, D = Action.C, Action.D
Interaction = List[Tuple[Action, Action]]


@dataclass(frozen=True)
class PlayerOutcome:
    wins: int
    mean_score_per_turn: float
    total_score: float
    matches: int


def _char_to_action(ch: str) -> Action:
    return C if ch.upper() == "C" else D


def load_player_matches(interactions_csv: str) -> Dict[str, List[Interaction]]:
    """Return {player_name: [ [(mine,theirs), ...], ...one list per match... ]}.

    Self-matches (a strategy against a copy of itself) are skipped so they don't
    distort the fingerprint.
    """
    # Group rows by interaction index, keep both perspectives.
    by_index: Dict[str, list] = defaultdict(list)
    with open(interactions_csv) as fh:
        for row in csv.DictReader(fh):
            by_index[row["Interaction index"]].append(row)

    out: Dict[str, List[Interaction]] = defaultdict(list)
    for _idx, rows in by_index.items():
        # Each interaction index should have two rows (the two perspectives),
        # except self-matches which we skip.
        perspectives = {(r["Player index"], r["Opponent index"]): r for r in rows}
        for (pi, oi), row in perspectives.items():
            if pi == oi:
                continue  # self-match
            mine = row["Actions"]
            opp_row = perspectives.get((oi, pi))
            if opp_row is None:
                continue
            theirs = opp_row["Actions"]
            pairs = [
                (_char_to_action(a), _char_to_action(b))
                for a, b in zip(mine, theirs)
            ]
            out[row["Player name"]].append(pairs)
    return dict(out)


def load_player_outcomes(interactions_csv: str) -> Dict[str, PlayerOutcome]:
    """Aggregate tournament wins and mean score-per-turn per player.

    Self-matches are excluded so totals match head-to-head round-robin play.
    """
    totals: Dict[str, dict] = {}
    with open(interactions_csv) as fh:
        for row in csv.DictReader(fh):
            if row["Player index"] == row["Opponent index"]:
                continue
            name = row["Player name"]
            bucket = totals.setdefault(
                name, {"wins": 0, "score_per_turn": [], "total_score": 0.0, "matches": 0}
            )
            bucket["wins"] += int(row["Win"])
            bucket["score_per_turn"].append(float(row["Score per turn"]))
            bucket["total_score"] += float(row["Score"])
            bucket["matches"] += 1

    return {
        name: PlayerOutcome(
            wins=data["wins"],
            mean_score_per_turn=(
                sum(data["score_per_turn"]) / len(data["score_per_turn"])
                if data["score_per_turn"]
                else float("nan")
            ),
            total_score=data["total_score"],
            matches=data["matches"],
        )
        for name, data in totals.items()
    }
