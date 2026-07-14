"""Export tournament CSVs into a JSON bundle for the interactive web report.

Reads results/data/*.csv (no tournament re-run) and writes:
    results/web/report.json

Optionally copies into the alair-website public path:

    python scripts/export_web_report.py \\
        --copy-to ../alair-website/public/rabbit-holes/llm-axelrod-tournament/report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_ipd.llm_player import display_llm_name, parse_llm_player

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "results", "data")
WEB = os.path.join(ROOT, "results", "web")

FINGERPRINT_KEYS = (
    "cooperation_rate",
    "niceness",
    "retaliation",
    "forgiveness",
    "provocability",
)

MODEL_COLORS = {
    "gpt-4o-mini": "#218BFF",
    "claude-haiku-4-5": "#A371F7",
}
CLASSIC_COLOR = "#8B949E"
FALLBACK_COLORS = ("#FF7B72", "#3FB950", "#D29922", "#F778BA", "#79C0FF")


def _load_csv_dicts(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _load_cooperation_matrix(path: str) -> dict:
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)[1:]
        matrix = {}
        for row in reader:
            matrix[row[0]] = {
                col: float(v) for col, v in zip(header, row[1:])
            }
        return {"players": header, "values": matrix}


def _parse_player(name: str) -> dict:
    parsed = parse_llm_player(name)
    if parsed:
        model, persona = parsed
        return {
            "id": name,
            "kind": "llm",
            "model": model,
            "persona": persona,
            "label": display_llm_name(name),
            "shortLabel": f"{model} · {persona}",
        }
    return {
        "id": name,
        "kind": "classic",
        "model": None,
        "persona": None,
        "label": name,
        "shortLabel": name,
    }


def _fingerprint_vector(fp: dict[str, float]) -> list[float]:
    return [float(fp[k]) for k in FINGERPRINT_KEYS]


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _nearest_classics(
    fingerprints: dict[str, dict[str, float]],
    players: list[dict],
) -> dict[str, dict]:
    classics = [p for p in players if p["kind"] == "classic"]
    classic_vecs = {
        p["id"]: _fingerprint_vector(fingerprints[p["id"]]) for p in classics
    }
    out = {}
    for p in players:
        if p["kind"] != "llm":
            continue
        vec = _fingerprint_vector(fingerprints[p["id"]])
        ranked = sorted(
            (
                {
                    "id": cid,
                    "label": next(c["label"] for c in classics if c["id"] == cid),
                    "distance": round(_euclidean(vec, cvec), 4),
                }
                for cid, cvec in classic_vecs.items()
            ),
            key=lambda row: row["distance"],
        )
        out[p["id"]] = {
            "nearest": ranked[0],
            "ranking": ranked,
        }
    return out


def _model_color_map(players: list[dict]) -> dict[str, str]:
    models: list[str] = []
    for p in players:
        if p["kind"] == "llm" and p["model"] not in models:
            models.append(p["model"])
    colors = {}
    for i, model in enumerate(models):
        colors[model] = MODEL_COLORS.get(
            model, FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
        )
    return colors


def build_report() -> dict:
    fp_rows = _load_csv_dicts(os.path.join(DATA, "fingerprints.csv"))
    outcome_rows = _load_csv_dicts(os.path.join(DATA, "outcomes.csv"))
    ranked_rows = _load_csv_dicts(os.path.join(DATA, "ranked_scores.csv"))

    fingerprints = {
        row["player"]: {k: float(row[k]) for k in FINGERPRINT_KEYS}
        for row in fp_rows
    }
    outcomes = {
        row["player"]: {
            "wins": int(row["wins"]),
            "meanScorePerTurn": float(row["mean_score_per_turn"]),
            "totalScore": float(row["total_score"]),
            "matches": int(row["matches"]),
        }
        for row in outcome_rows
    }
    ranked = {
        row["player"]: {
            "rank": int(row["rank"]),
            "meanNormalisedScore": float(row["mean_normalised_score"]),
        }
        for row in ranked_rows
    }

    # Preserve fingerprint file order as canonical player order.
    order = [row["player"] for row in fp_rows]
    players = [_parse_player(name) for name in order]
    model_colors = _model_color_map(players)

    for p in players:
        p["color"] = (
            model_colors[p["model"]] if p["kind"] == "llm" else CLASSIC_COLOR
        )
        p["fingerprint"] = fingerprints[p["id"]]
        p["outcomes"] = outcomes[p["id"]]
        p["ranked"] = ranked.get(p["id"])

    nearest = _nearest_classics(fingerprints, players)
    for p in players:
        if p["id"] in nearest:
            p["nearestClassic"] = nearest[p["id"]]["nearest"]
            p["nearestClassicRanking"] = nearest[p["id"]]["ranking"]

    # Series: connect persona variants of the same model (Cursor-style).
    series = []
    by_model: dict[str, list[str]] = {}
    for p in players:
        if p["kind"] == "llm":
            by_model.setdefault(p["model"], []).append(p["id"])
    persona_rank = {
        "cooperative": 0,
        "neutral": 1,
        "payoff_only": 2,
        "selfish": 3,
    }
    for model, ids in by_model.items():
        ids_sorted = sorted(
            ids,
            key=lambda pid: persona_rank.get(
                next(p["persona"] for p in players if p["id"] == pid), 99
            ),
        )
        series.append(
            {
                "id": model,
                "label": model,
                "color": model_colors[model],
                "playerIds": ids_sorted,
            }
        )

    matches = next(iter(outcomes.values()))["matches"] if outcomes else None
    turns = None
    interactions_path = os.path.join(DATA, "interactions_raw.csv")
    if os.path.exists(interactions_path):
        with open(interactions_path, newline="") as fh:
            first = next(csv.DictReader(fh), None)
            if first:
                turns = int(first["Turns"])

    return {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "title": "LLM IPD Fingerprints",
        "subtitle": (
            "Behavioral fingerprints of language models in the iterated "
            "prisoner's dilemma, compared against classic Axelrod strategies."
        ),
        "meta": {
            "players": len(players),
            "classics": sum(1 for p in players if p["kind"] == "classic"),
            "llms": sum(1 for p in players if p["kind"] == "llm"),
            "matchesPerPlayer": matches,
            "turns": turns,
            "fingerprintDimensions": list(FINGERPRINT_KEYS),
            "defaultAxes": {
                "x": "cooperation_rate",
                "y": "forgiveness",
            },
        },
        "metrics": [
            {
                "id": "cooperation_rate",
                "label": "Cooperation",
                "shortLabel": "Coop.",
                "description": "Fraction of C moves overall.",
                "domain": [0, 1],
            },
            {
                "id": "niceness",
                "label": "Niceness",
                "shortLabel": "Nice",
                "description": "Fraction of matches with no first defection.",
                "domain": [0, 1],
            },
            {
                "id": "retaliation",
                "label": "Retaliation",
                "shortLabel": "Retal.",
                "description": "P(defect next | opponent just defected).",
                "domain": [0, 1],
            },
            {
                "id": "forgiveness",
                "label": "Forgiveness",
                "shortLabel": "Forgive",
                "description": "P(cooperate next | opponent returned to C).",
                "domain": [0, 1],
            },
            {
                "id": "provocability",
                "label": "Provocability",
                "shortLabel": "Prov.",
                "description": "Retaliation to the opponent's first defection.",
                "domain": [0, 1],
            },
            {
                "id": "mean_score_per_turn",
                "label": "Score / turn",
                "shortLabel": "Score",
                "description": "Mean payoff per turn across all matches.",
                "domain": "auto",
            },
        ],
        "players": players,
        "series": series,
        "cooperationMatrix": _load_cooperation_matrix(
            os.path.join(DATA, "cooperation_matrix.csv")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(WEB, "report.json"),
        help="Primary JSON output path",
    )
    parser.add_argument(
        "--copy-to",
        action="append",
        default=[],
        help="Additional path(s) to copy the report into (repeatable)",
    )
    args = parser.parse_args()

    report = build_report()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"Wrote {args.out} ({len(report['players'])} players)")

    for dest in args.copy_to:
        os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
        shutil.copyfile(args.out, dest)
        print(f"Copied → {dest}")


if __name__ == "__main__":
    main()
