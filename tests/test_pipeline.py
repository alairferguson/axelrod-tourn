"""Tests that run with NO API keys.

We monkeypatch litellm.completion so LLMPlayer behaves like a deterministic
scripted strategy, then verify: parsing, the fairness invariants, fingerprint
metrics on known interaction sequences, and an end-to-end tournament + analysis.
"""

import os
import sys

import axelrod as axl
from axelrod.action import Action

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_ipd.llm_player import LLMPlayer  # noqa: E402
from llm_ipd.fingerprint import (  # noqa: E402
    cooperation_rate, niceness, retaliation, forgiveness, provocability,
    probe_profile, profile_distance,
)
from llm_ipd.roster import classic_factories  # noqa: E402

C, D = Action.C, Action.D


def test_parse():
    assert LLMPlayer._parse("C") == C
    assert LLMPlayer._parse("D") == D
    assert LLMPlayer._parse("I will Defect.") == D
    assert LLMPlayer._parse("cooperate") == C
    assert LLMPlayer._parse("A", persona="payoff_only") == C
    assert LLMPlayer._parse("B", persona="payoff_only") == D
    assert LLMPlayer._parse("C", persona="payoff_only") is None
    assert LLMPlayer._parse("") is None
    assert LLMPlayer._parse(None) is None
    print("test_parse OK")


def test_payoff_only_prompt():
    from llm_ipd.prompts import build_prompt, PERSONA_PROMPTS

    prompt = build_prompt([], [], axl.Game(), persona="payoff_only")
    lower = prompt.lower()
    assert "cooperat" not in lower
    assert "defect" not in lower
    assert "(A, A)" in prompt
    assert "ONLY A or B" in prompt
    print("test_payoff_only_prompt OK")


def test_persona_output_rules_consistent():
    from llm_ipd.prompts import PERSONA_PROMPTS

    cd_rules = (
        "Output rules (strict):\n"
        "  - Reply with exactly one character: C or D\n"
        "  - No explanation, punctuation, or other text"
    )
    ab_rules = (
        "Output rules (strict):\n"
        "  - Reply with exactly one character: A or B\n"
        "  - No explanation, punctuation, or other text"
    )
    for persona in ("neutral", "selfish", "cooperative"):
        assert PERSONA_PROMPTS[persona].endswith(cd_rules), persona
        assert "Reason" not in PERSONA_PROMPTS[persona]
    assert PERSONA_PROMPTS["payoff_only"].endswith(ab_rules)
    assert "Reason" not in PERSONA_PROMPTS["payoff_only"]
    print("test_persona_output_rules_consistent OK")


def test_fingerprint_known_sequences():
    # Always-defect-against-cooperator: never nice, full coop = 0
    m = [[(D, C), (D, C), (D, C)]]
    assert cooperation_rate(m) == 0.0
    assert niceness(m) == 0.0

    # Pure cooperator: nice, coop = 1
    m = [[(C, C), (C, C)]]
    assert cooperation_rate(m) == 1.0
    assert niceness(m) == 1.0

    # Tit-for-tat-like: opponent defects at t1, player retaliates at t2
    m = [[(C, C), (C, D), (D, C), (C, C)]]
    # opponent defected at index1 -> player's next (index2) is D => retaliation=1
    assert retaliation(m) == 1.0
    # opponent returned to C at index2 (after D at index1); player's next
    # (index3) is C => forgiveness = 1
    assert forgiveness(m) == 1.0
    # first opponent defection at index1 -> player next D => provocability 1
    assert provocability(m) == 1.0
    print("test_fingerprint_known_sequences OK")


def test_classic_probe_profiles_distinct():
    profs = {n: probe_profile(f) for n, f in classic_factories().items()}
    # Cooperator and Defector must be maximally different
    d = profile_distance(profs["Cooperator"], profs["Defector"])
    assert d == 1.0, d
    # Tit For Tat should differ from Cooperator (it punishes defections)
    assert profile_distance(profs["Tit For Tat"], profs["Cooperator"]) > 0
    print("test_classic_probe_profiles_distinct OK")


def test_load_player_outcomes(tmp_path=None):
    import csv
    import os
    import tempfile

    from llm_ipd.io_utils import load_player_outcomes  # noqa: E402

    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_path, "interactions.csv")
    rows = [
        {
            "Interaction index": "0", "Player index": "0", "Opponent index": "1",
            "Player name": "Cooperator", "Opponent name": "Defector",
            "Win": "0", "Score per turn": "0.0", "Score": "0",
        },
        {
            "Interaction index": "0", "Player index": "1", "Opponent index": "0",
            "Player name": "Defector", "Opponent name": "Cooperator",
            "Win": "1", "Score per turn": "5.0", "Score": "15",
        },
        {
            "Interaction index": "1", "Player index": "0", "Opponent index": "0",
            "Player name": "Cooperator", "Opponent name": "Cooperator",
            "Win": "0", "Score per turn": "3.0", "Score": "9",
        },
        {
            "Interaction index": "2", "Player index": "0", "Opponent index": "2",
            "Player name": "LLM:gpt-4o-mini[cooperative]",
            "Opponent name": "Defector",
            "Win": "0", "Score per turn": "1.5", "Score": "6",
        },
        {
            "Interaction index": "2", "Player index": "2", "Opponent index": "0",
            "Player name": "Defector", "Opponent name": "LLM:gpt-4o-mini[cooperative]",
            "Win": "1", "Score per turn": "4.5", "Score": "18",
        },
    ]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    outcomes = load_player_outcomes(csv_path)
    assert outcomes["Cooperator"].wins == 0
    assert outcomes["Cooperator"].matches == 1
    assert outcomes["Cooperator"].mean_score_per_turn == 0.0
    assert outcomes["Defector"].wins == 2
    assert outcomes["Defector"].mean_score_per_turn == 4.75
    assert outcomes["Defector"].total_score == 33.0
    assert outcomes["LLM:gpt-4o-mini[cooperative]"].mean_score_per_turn == 1.5
    assert outcomes["LLM:gpt-4o-mini[cooperative]"].total_score == 6.0
    print("test_load_player_outcomes OK")


def test_llm_player_mocked_end_to_end(monkeypatch=None):
    # Make LLMPlayer deterministically "always cooperate" by patching litellm.
    import types
    fake = types.SimpleNamespace()

    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": "C"}}]}

    import importlib
    litellm = importlib.import_module("litellm")
    orig = litellm.completion
    litellm.completion = fake_completion
    try:
        llm = LLMPlayer(model="fake/model", cache=None)
        # Plays C against a defector every round => gets exploited, stays nice
        match = axl.Match([llm, axl.Defector()], turns=4)
        result = match.play()
        my_moves = [str(mine) for (mine, _t) in result]
        assert my_moves == ["C", "C", "C", "C"], my_moves
        assert llm.n_calls > 0
    finally:
        litellm.completion = orig
    print("test_llm_player_mocked_end_to_end OK")


if __name__ == "__main__":
    test_parse()
    test_payoff_only_prompt()
    test_persona_output_rules_consistent()
    test_fingerprint_known_sequences()
    test_classic_probe_profiles_distinct()
    test_load_player_outcomes()
    test_llm_player_mocked_end_to_end()
    print("\nAll tests passed.")
