"""The curated reference frame of classic strategies.

These are the "known landmarks" we fingerprint the LLM players against. Kept
small and interpretable on purpose -- each one occupies a recognisable corner of
strategy space (nice vs nasty, forgiving vs grudging, reactive vs fixed).
"""

from __future__ import annotations

import axelrod as axl


def classic_roster() -> list[axl.Player]:
    return [
        axl.TitForTat(),          # nice, retaliatory, forgiving -- the benchmark
        axl.Grudger(),        # nice but unforgiving (a.k.a. Grudger)
        axl.WinStayLoseShift(),   # Pavlov: repeat if it paid off, switch if not
        axl.GTFT(),               # Generous TFT: forgives some defections
        axl.Cooperator(),         # always C
        axl.Defector(),           # always D
        axl.Random(),             # 50/50 baseline
    ]


def classic_factories() -> dict:
    """Name -> zero-arg factory returning a fresh instance, for probe profiling."""
    return {
        "Tit For Tat": axl.TitForTat,
        "Grim Trigger": axl.Grudger,
        "Win-Stay-Lose-Shift": axl.WinStayLoseShift,
        "Generous Tit For Tat": axl.GTFT,
        "Cooperator": axl.Cooperator,
        "Defector": axl.Defector,
        "Random": axl.Random,
    }
