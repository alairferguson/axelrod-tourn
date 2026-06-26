# Behavioral Fingerprinting of LLMs in the Iterated Prisoner's Dilemma

**Research question.** When a language model plays the iterated prisoner's
dilemma (IPD), *how does it actually play*? Which classic strategy does its
behavior most resemble, and where does it sit on the dimensions Robert Axelrod
identified as decisive — niceness, forgiveness, retaliation, and provocability?
Winning the tournament is secondary; the **behavioral profile** is the result.

This reframes the now-common "drop an LLM into the Axelrod tournament" demo from
a stunt ("can it win?") into a measurement ("what *is* it, behaviorally?").

---

## What this does

1. Runs a round-robin Axelrod tournament with a curated set of **classic
   strategies** as a reference frame plus one or more **LLM players** (any model
   reachable through [`litellm`](https://github.com/BerriAI/litellm)).
2. Computes a **behavioral fingerprint** for every player from the recorded
   move histories.
3. Finds each LLM's **nearest classic strategy** using a standardized probe
   battery.
4. Draws the **hero figure**: every player placed in strategy space, classics as
   dots, LLMs as stars, so you can read off "this model plays like X" at a
   glance.

## Fairness: how the LLM stays on equal footing with the classics

A reasonable worry: classic strategies are "locked in" before the tournament, so
does letting an LLM adapt give it an unfair edge? Within-match adaptation is not
an advantage — it *is* the game. Tit-for-Tat adapts every round. The things that
*would* be unfair are different, and this project rules them out by construction:

- **Per-match statelessness.** The model is given only the current match's
  history. It never carries memory across matches. (`LLMPlayer` keeps no state
  outside Axelrod's per-match `history`.)
- **Moves-only information.** The prompt contains the payoff matrix and the
  sequence of past moves — exactly what a classic strategy sees. The model is
  **never** told the opponent's identity or strategy name.
- **Determinism by default.** `temperature=0` for the main runs, so results are
  reproducible. Prompt-paraphrase and temperature sensitivity are studied
  deliberately as a side experiment, not left as uncontrolled noise.

See the docstring in `src/llm_ipd/llm_player.py` for the enforcement details.

## The fingerprint dimensions

| Dimension | Definition |
|---|---|
| **cooperation_rate** | fraction of C moves overall |
| **niceness** | fraction of matches where the player never defected *first* |
| **retaliation** | P(player defects next \| opponent just defected) |
| **forgiveness** | P(player cooperates next \| opponent just returned to C) |
| **provocability** | retaliation to the opponent's *first* defection only |

"Nearest classic strategy" uses the **probe method**: every strategy is replayed
against the same standardized battery of opponent behaviors (all-C, all-D, a
single defection, alternating, defect-then-repent, …) and we compare the
resulting move profiles with a normalized Hamming distance. This controls for the
fact that different opponents naturally elicit different histories.

## Install

```bash
pip install -r requirements.txt        # or: pip install -e .
cp .env.example .env                    # add API keys for the providers you use
```

No API key is needed to run the classic-only tournament or the test suite.

## Run

```bash
# Classic strategies only (no API calls, free):
python -m llm_ipd.tournament

# Add LLM players (needs the matching API key in your environment):
python -m llm_ipd.tournament --models gpt-4o-mini --turns 30 --repetitions 5

# Multiple models at once:
python -m llm_ipd.tournament --models gpt-4o-mini claude-haiku-4-5 ollama/llama3

# Analyze + draw figures (re-pass the same --models):
python -m llm_ipd.analyze --models gpt-4o-mini
```

Run from the `src/` directory or install the package first so `llm_ipd` is
importable (`pip install -e .`).

### The persona knob (stretch experiment)

Re-run with a different system prompt to see how disposition shifts the
fingerprint of the *same* model:

```bash
python -m llm_ipd.tournament --models gpt-4o-mini --persona selfish
python -m llm_ipd.tournament --models gpt-4o-mini --persona cooperative
```

Personas are defined in `src/llm_ipd/prompts.py`.

## Cost & latency

LLM players make real API calls — a full round-robin can be thousands of calls.
Mitigations built in: a persistent on-disk **response cache** (identical game
states recur constantly), small default model choices, and modest
`--turns` / `--repetitions`. Estimate your call budget before a big run.

## Outputs

```
results/data/ranked_scores.csv        final ranking + normalised scores
results/data/cooperation_matrix.csv   who cooperates against whom
results/data/fingerprints.csv         the five fingerprint metrics per player
results/data/interactions_raw.csv     every match's move sequence (source of truth)
results/data/llm_cache.json           cached model responses
results/figures/fingerprint_space.png the hero figure
```

## Project layout

```
src/llm_ipd/
  llm_player.py   the LLMPlayer Axelrod strategy (the only bespoke surface)
  prompts.py      moves-only prompt construction + persona system prompts
  cache.py        persistent response cache
  roster.py       curated classic reference strategies
  tournament.py   runner: build players, play, save results
  io_utils.py     reconstruct (mine, theirs) move pairs from the CSV
  fingerprint.py  the fingerprint metrics + probe / nearest-strategy logic
  analyze.py      compute fingerprints, nearest strategy, draw the hero figure
tests/
  test_pipeline.py  runs with NO API keys (litellm is mocked)
```

## Tests

```bash
python tests/test_pipeline.py
```

## Scope (deliberate cuts)

Kept out to stay finishable, and worth naming as future work:

- **Fine-tuning / LoRA / TRL.** The disposition question is answered here via
  prompting (the persona knob) instead of gradient updates — no GPU, no training
  infrastructure.
- **Leakage into unrelated tasks.** A different project.
- **Full (T, S) game-grid sweep** across game types (Stag Hunt, Chicken,
  Harmony). A natural v2: re-run the fingerprint as you move the payoffs across
  game-type boundaries.

## Known refinements

- Label de-overlap on the hero figure when many players cluster (e.g. the nice/
  forgiving corner). A repulsion layout or `adjustText` would clean it up.
- An LLM may not be a single deterministic strategy in Axelrod's sense; `t=0`
  plus the paraphrase side-experiment is how this project addresses that.

## Credits

Built on the [Axelrod-Python](https://github.com/Axelrod-Python/Axelrod)
library (Knight et al., *Journal of Open Research Software*, 2016).
