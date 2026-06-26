"""Behavioral fingerprinting of a player from its recorded match interactions.

Given the per-match interaction histories Axelrod records, we compute the
dimensions Axelrod's own analysis found decisive:

  - cooperation_rate : overall fraction of C moves
  - niceness         : fraction of matches in which the player never defected
                       first (defected only after the opponent did, or never)
  - retaliation      : P(player defects next | opponent just defected)
  - forgiveness      : P(player cooperates next | opponent just returned to C
                       after a defection) -- how readily it rebuilds cooperation
  - provocability    : retaliation restricted to the FIRST opponent defection
                       (does a single defection trigger a response at all)

Then `nearest_strategy` compares a target player's response profile to each
classic strategy on a STANDARDIZED battery of histories (the "probe" method):
every player is asked the same set of canonical situations, and we compare their
move distributions. This is cleaner than comparing on whatever histories happened
to occur, because it controls for the fact that different opponents elicit
different histories.

Interactions format (Axelrod's native form): a list of (my_move, their_move)
tuples per match. We pass in {match_label: [(C,C), (D,C), ...], ...}.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import axelrod as axl
from axelrod.action import Action

C, D = Action.C, Action.D
Interaction = List[Tuple[Action, Action]]


# --- Single-player fingerprint -------------------------------------------------

def cooperation_rate(matches: Sequence[Interaction]) -> float:
    moves = [mine for m in matches for (mine, _) in m]
    if not moves:
        return float("nan")
    return sum(1 for x in moves if x == C) / len(moves)


def niceness(matches: Sequence[Interaction]) -> float:
    """Fraction of matches where the player never defected before the opponent."""
    nice = 0
    for m in matches:
        defected_first = False
        for mine, theirs in m:
            if mine == D and theirs == C and _all_prior_coop(m, mine, theirs):
                pass  # handled below
        # Simpler: walk until someone defects.
        opp_defected = False
        player_nice = True
        for mine, theirs in m:
            if mine == D and not opp_defected:
                player_nice = False
                break
            if theirs == D:
                opp_defected = True
        nice += int(player_nice)
    return nice / len(matches) if matches else float("nan")


def _all_prior_coop(*_args) -> bool:  # retained for readability of niceness
    return True


def _conditional(matches: Sequence[Interaction], condition) -> float:
    """P(player's next move | condition on the round just played).
    `condition` maps (prev_mine, prev_theirs, idx, match) -> bool|None.
    Returns fraction of qualifying rounds where the player's NEXT move is D for
    retaliation-style, or C for forgiveness-style; the caller picks via the
    `target` returned. We standardize: condition returns the target action or
    None to skip."""
    hits = 0
    total = 0
    for m in matches:
        for i in range(len(m) - 1):
            target = condition(m, i)
            if target is None:
                continue
            total += 1
            next_mine = m[i + 1][0]
            hits += int(next_mine == target)
    return hits / total if total else float("nan")


def retaliation(matches: Sequence[Interaction]) -> float:
    """P(I defect next | opponent defected this round)."""
    def cond(m, i):
        _, theirs = m[i]
        return D if theirs == D else None
    return _conditional(matches, cond)


def forgiveness(matches: Sequence[Interaction]) -> float:
    """P(I cooperate next | opponent just switched back to C after defecting)."""
    def cond(m, i):
        if i == 0:
            return None
        prev_theirs = m[i - 1][1]
        _, theirs = m[i]
        if prev_theirs == D and theirs == C:
            return C
        return None
    return _conditional(matches, cond)


def provocability(matches: Sequence[Interaction]) -> float:
    """P(I defect next | this is the opponent's FIRST defection in the match)."""
    def cond(m, i):
        _, theirs = m[i]
        if theirs != D:
            return None
        # first defection only
        if any(t == D for (_, t) in m[:i]):
            return None
        return D
    return _conditional(matches, cond)


def fingerprint(matches: Sequence[Interaction]) -> Dict[str, float]:
    return {
        "cooperation_rate": cooperation_rate(matches),
        "niceness": niceness(matches),
        "retaliation": retaliation(matches),
        "forgiveness": forgiveness(matches),
        "provocability": provocability(matches),
    }


# --- Nearest classic strategy (probe method) -----------------------------------

# Standardized battery of opponent behaviors to probe every strategy with.
# Each value is a fixed cycle of OPPONENT moves; we replay each strategy against
# it using a real Axelrod Match (no manual history management).
PROBE_PATTERNS: Dict[str, str] = {
    "all_C": "C",
    "all_D": "D",
    "one_defection": "CCDCCC",
    "alternating": "CDCDCD",
    "defect_then_repent": "DDCCCC",
    "tit_for_tat_like": "CCCDCC",
}


def probe_profile(player_factory, turns: int = 6) -> Dict[str, str]:
    """Replay a strategy against each canonical opponent pattern using a real
    Axelrod Match, and record the player's move string.

    `player_factory` returns a fresh player instance, so state resets between
    probes. Returns {pattern_name: "CCDC..."}. The probed player is always
    position 0 in the match. For an LLM player each probe round is one API call,
    but cached, so re-runs are cheap.
    """
    profiles: Dict[str, str] = {}
    for name, cycle in PROBE_PATTERNS.items():
        player = player_factory()
        opponent = axl.Cycler(cycle=cycle * (turns // len(cycle) + 1))
        match = axl.Match([player, opponent], turns=turns)
        interactions = match.play()
        profiles[name] = "".join(str(mine) for (mine, _theirs) in interactions)
    return profiles


def profile_distance(a: Dict[str, str], b: Dict[str, str]) -> float:
    """Normalized Hamming distance between two probe profiles (0 = identical)."""
    total = 0
    diff = 0
    for key in a:
        sa, sb = a[key], b.get(key, "")
        for ca, cb in zip(sa, sb):
            total += 1
            diff += int(ca != cb)
    return diff / total if total else float("nan")


def nearest_strategy(target_profile: Dict[str, str],
                     classic_profiles: Dict[str, Dict[str, str]]
                     ) -> List[Tuple[str, float]]:
    """Rank classic strategies by closeness to the target's probe profile.
    Returns [(strategy_name, distance), ...] sorted nearest first."""
    ranked = [
        (name, profile_distance(target_profile, prof))
        for name, prof in classic_profiles.items()
    ]
    ranked.sort(key=lambda x: x[1])
    return ranked
