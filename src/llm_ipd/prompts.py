"""Prompt construction for LLMPlayer.

The user prompt serializes ONLY the payoff matrix and the move history of the
current match -- the same information a classic Axelrod strategy can see. It
never names the opponent or its strategy (fairness invariant #2).

The system prompts double as the "disposition knob" for the persona side
experiment: same model, different system prompt, measure how the behavioral
fingerprint shifts.
"""

from __future__ import annotations

import axelrod as axl
from axelrod.action import Action

C, D = Action.C, Action.D

# --- System prompts (personas) -------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are playing a repeated game. Each round you choose to either "
    "Cooperate (C) or Defect (D). You will be shown the payoff rules and the "
    "history of play so far. Reason if you wish, but your final answer must be "
    "exactly one character: C or D."
)

PERSONA_PROMPTS = {
    "neutral": DEFAULT_SYSTEM_PROMPT,
    "selfish": (
        "You are a ruthless, self-interested agent. Your only goal is to "
        "maximize your own score, with no regard for the other player. Each "
        "round choose Cooperate (C) or Defect (D). Answer with exactly one "
        "character: C or D."
    ),
    "cooperative": (
        "You are a fair-minded agent who values mutual benefit and long-term "
        "trust. Each round choose Cooperate (C) or Defect (D). Answer with "
        "exactly one character: C or D."
    ),
}


# --- User prompt (game state) --------------------------------------------------

def _payoff_block(game: axl.Game) -> str:
    """Render the payoff matrix in plain language from the live Game object.
    RPST = Reward (CC), Punishment (DD), Sucker (CD), Temptation (DC)."""
    R, P, S, T = game.RPST()
    return (
        "Payoffs each round (your_move, their_move) -> your_points:\n"
        f"  (C, C) -> {R}    (you cooperate, they cooperate)\n"
        f"  (C, D) -> {S}    (you cooperate, they defect)\n"
        f"  (D, C) -> {T}    (you defect, they cooperate)\n"
        f"  (D, D) -> {P}    (you defect, they defect)\n"
    )


def _history_block(my_history, opponent_history) -> str:
    if len(my_history) == 0:
        return "No rounds have been played yet. This is round 1."
    lines = []
    for i, (mine, theirs) in enumerate(zip(my_history, opponent_history), start=1):
        lines.append(f"  Round {i}: you played {mine}, they played {theirs}")
    return "History so far:\n" + "\n".join(lines)


def build_prompt(my_history, opponent_history, game: axl.Game) -> str:
    return (
        _payoff_block(game)
        + "\n"
        + _history_block(my_history, opponent_history)
        + "\n\nWhat do you play this round? Answer with exactly one character: "
        "C or D."
    )
