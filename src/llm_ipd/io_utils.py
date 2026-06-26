"""Reconstruct per-player match histories as (mine, theirs) Action pairs from the
tournament interactions CSV. Both player perspectives share an Interaction index,
so we join them to recover full move pairs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List, Tuple

from axelrod.action import Action

C, D = Action.C, Action.D
Interaction = List[Tuple[Action, Action]]


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
