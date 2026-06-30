"""Prompt construction for LLMPlayer.

The user prompt serializes ONLY the payoff matrix and the move history of the
current match -- the same information a classic Axelrod strategy can see. It
never names the opponent or its strategy (fairness invariant #2).

The system prompts double as the "disposition knob" for the persona side
experiment: same model, different system prompt, measure how the behavioral
fingerprint shifts. Output-format instructions are identical across personas;
only the disposition paragraph varies (except payoff_only, which uses A/B labels).
"""

from __future__ import annotations

import axelrod as axl
from axelrod.action import Action

C, D = Action.C, Action.D

# --- Shared output rules (identical across personas) ----------------------------

_OUTPUT_RULES_CD = (
    "Output rules (strict):\n"
    "  - Reply with exactly one character: C or D\n"
    "  - No explanation, punctuation, or other text"
)

_OUTPUT_RULES_AB = (
    "Output rules (strict):\n"
    "  - Reply with exactly one character: A or B\n"
    "  - No explanation, punctuation, or other text"
)

_USER_CLOSING_CD = (
    "\n\nYour move this round (reply with ONLY C or D, no other text):"
)

_USER_CLOSING_AB = (
    "\n\nYour move this round (reply with ONLY A or B, no other text):"
)

# --- Persona dispositions (the controlled variation) ----------------------------

_PERSONA_DISPOSITION = {
    "neutral": (
        "You are playing a repeated Prisoner's Dilemma. Each round choose "
        "Cooperate (C) or Defect (D). You will see the payoff table and this "
        "match's move history."
    ),
    "selfish": (
        "You are a ruthless, self-interested agent. Your only goal is to "
        "maximize your own score, with no regard for the other player. Each "
        "round choose Cooperate (C) or Defect (D). You will see the payoff "
        "table and this match's move history."
    ),
    "cooperative": (
        "You are a fair-minded agent who values mutual benefit and long-term "
        "trust. Each round choose Cooperate (C) or Defect (D). You will see "
        "the payoff table and this match's move history."
    ),
    "payoff_only": (
        "You are playing a repeated two-player game. Each round you choose "
        "action A or action B. You will see a payoff table and this match's "
        "choice history. Maximize your own total points."
    ),
}


def _build_system_prompt(persona: str) -> str:
    disposition = _PERSONA_DISPOSITION[persona]
    if persona == "payoff_only":
        return f"{disposition}\n{_OUTPUT_RULES_AB}"
    return f"{disposition}\n{_OUTPUT_RULES_CD}"


PERSONA_PROMPTS = {name: _build_system_prompt(name) for name in _PERSONA_DISPOSITION}
DEFAULT_SYSTEM_PROMPT = PERSONA_PROMPTS["neutral"]


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


def _payoff_block_util(game: axl.Game) -> str:
    """Payoff table with no cooperate/defect framing — numbers only."""
    R, P, S, T = game.RPST()
    return (
        "Payoffs each round (your choice, their choice) -> your_points:\n"
        f"  (A, A) -> {R}\n"
        f"  (A, B) -> {S}\n"
        f"  (B, A) -> {T}\n"
        f"  (B, B) -> {P}\n"
    )


def _history_block(my_history, opponent_history) -> str:
    if len(my_history) == 0:
        return "No rounds have been played yet. This is round 1."
    lines = []
    for i, (mine, theirs) in enumerate(zip(my_history, opponent_history), start=1):
        lines.append(f"  Round {i}: you played {mine}, they played {theirs}")
    return "History so far:\n" + "\n".join(lines)


def _history_block_util(my_history, opponent_history) -> str:
    if len(my_history) == 0:
        return "No rounds have been played yet. This is round 1."

    def _label(action):
        return "A" if action == C else "B"

    lines = []
    for i, (mine, theirs) in enumerate(zip(my_history, opponent_history), start=1):
        lines.append(
            f"  Round {i}: you chose {_label(mine)}, they chose {_label(theirs)}"
        )
    return "History so far:\n" + "\n".join(lines)


def build_prompt(my_history, opponent_history, game: axl.Game,
                 persona: str = "neutral") -> str:
    if persona == "payoff_only":
        return (
            _payoff_block_util(game)
            + "\n"
            + _history_block_util(my_history, opponent_history)
            + _USER_CLOSING_AB
        )
    return (
        _payoff_block(game)
        + "\n"
        + _history_block(my_history, opponent_history)
        + _USER_CLOSING_CD
    )
