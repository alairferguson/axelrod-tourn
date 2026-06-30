"""LLMPlayer: an Axelrod-compatible strategy that decides its move by querying
a language model through litellm.

Design invariants (these keep the LLM on equal footing with classic strategies;
see README "Fairness"):

  1. PER-MATCH STATELESSNESS. The model is given ONLY the current match's move
     history. It never carries memory across matches. Axelrod resets self.history
     between matches, and we never store state outside of it, so this holds by
     construction.

  2. MOVES-ONLY INFORMATION. The prompt contains the payoff matrix and the
     sequence of past (my move, their move) pairs -- exactly what a classic
     strategy can see via self.history / opponent.history. It is NEVER told the
     opponent's identity or strategy name.

  3. DETERMINISM BY DEFAULT. temperature defaults to 0 so the main runs are
     reproducible. Prompt-paraphrase / temperature sensitivity is studied
     deliberately as a side experiment, not left as uncontrolled noise.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import axelrod as axl
from axelrod.action import Action

from .cache import ResponseCache
from .prompts import build_prompt, DEFAULT_SYSTEM_PROMPT

C, D = Action.C, Action.D


class LLMPlayer(axl.Player):
    """An Iterated Prisoner's Dilemma player backed by an LLM.

    Parameters
    ----------
    model:
        A litellm model string, e.g. "gpt-4o-mini", "claude-haiku-4-5",
        "ollama/llama3". This is the only thing that changes to swap providers.
    system_prompt:
        The persona / instruction block. Swapping this is the "disposition knob"
        (stretch experiment). Defaults to a neutral, rules-only prompt.
    temperature:
        Sampling temperature. 0.0 for reproducible main runs.
    cache:
        Optional ResponseCache. Identical (model, system, user) prompts recur
        constantly across repetitions; caching cuts cost and latency massively.
    name:
        Display name in tournament results. Defaults to "LLM: <model>".
    fallback:
        Move to play if the API call fails or the reply can't be parsed.
        Default C (the "nice" default); logged so you can audit how often it fires.
    """

    # Axelrod classifier. We mark this as stochastic unless temperature == 0,
    # and as using no special knowledge of the match/opponent.
    classifier = {
        "memory_depth": float("inf"),  # may condition on full history
        "stochastic": True,
        "long_run_time": True,         # network call per move
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        cache: Optional[ResponseCache] = None,
        name: Optional[str] = None,
        fallback: Action = C,
        persona: str = "neutral",
    ) -> None:
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt
        self.persona = persona
        self.temperature = temperature
        self.cache = cache
        self.fallback = fallback
        self.name = name or f"LLM: {model}"
        # Per-instance classifier copy so temperature can flip the stochastic flag.
        self.classifier = dict(self.classifier)
        self.classifier["stochastic"] = temperature > 0.0
        # Diagnostics, not used for decisions.
        self.n_calls = 0
        self.n_fallbacks = 0

    # Axelrod calls this once per turn. `opponent` exposes .history only.
    def strategy(self, opponent: axl.Player) -> Action:
        prompt = build_prompt(
            my_history=self.history,
            opponent_history=opponent.history,
            game=self.match_attributes.get("game", axl.Game()),
            persona=self.persona,
        )
        reply = self._query(prompt)
        move = self._parse(reply, self.persona)
        if move is None:
            self.n_fallbacks += 1
            return self.fallback
        return move

    def _query(self, user_prompt: str) -> Optional[str]:
        key = (self.model, self.system_prompt, self.temperature, user_prompt)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        # Imported lazily so the module imports fine with no API keys set
        # (e.g. for running the classic-only tournament or the test suite).
        import litellm

        try:
            self.n_calls += 1
            resp = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=1,
            )
            text = resp["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - we want any failure to fall back
            text = None
            self._last_error = repr(exc)

        if self.cache is not None and text is not None:
            self.cache.set(key, text)
        return text

    def __repr__(self) -> str:
        # Axelrod's default repr dumps all init_kwargs; we want the clean name
        # for tournament labels and figures.
        return self.name

    @staticmethod
    def _parse(reply: Optional[str], persona: str = "neutral") -> Optional[Action]:
        """Extract a single move from the model's reply. Robust to chatter
        like 'I will Defect.' Returns None if nothing parseable is found."""
        if not reply:
            return None
        text = reply.strip().upper()
        if persona == "payoff_only":
            if text and text[0] in ("A", "B"):
                return C if text[0] == "A" else D
            if "B" in text and "A" not in text:
                return D
            if "A" in text and "B" not in text:
                return C
            return None
        # Prefer a bare leading token.
        if text and text[0] in ("C", "D"):
            return C if text[0] == "C" else D
        # Otherwise look for the words.
        if re.search(r"\bDEFECT", text):
            return D
        if re.search(r"\bCOOPERAT", text):
            return C
        if "D" in text and "C" not in text:
            return D
        if "C" in text and "D" not in text:
            return C
        return None
