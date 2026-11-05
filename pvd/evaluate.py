"""Score every predictor (live model, three backtests, crowd) on the same matches.

The predictors are ordered by how much look-ahead bias they let in. Only the live model and
the crowd are truly out-of-sample; each backtest below them leaks a little more information
from after the match into its prediction.

Shared by scripts/make_results.py (tables) and scripts/make_charts.py (look-ahead chart).
"""
import csv

from pvd import RESULTS


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


matches = load(RESULTS / "matches.csv")
backtest = {(b["round"], b["match"]): b for b in load(RESULTS / "backtest_v8.csv")}
strict = {(b["round"], b["match"]): b for b in load(RESULTS / "backtest_strict_wf.csv")}

decided = [m for m in matches if m["winner"] in ("L", "R")]
scored = [m for m in decided if num(m["model_p_left"]) is not None]  # the matches with a live call

# (key, label, P(left) per match, look-ahead bias), from least to most leakage
PREDICTORS = [
    ("live", "**Live model**", lambda m: num(m["model_p_left"]), "None"),
    ("swf", "Strict walk-forward",
     lambda m: num(strict.get((m["round"], m["match"]), {}).get("strict_wf_p_left")), "Minimal"),
    ("wf", "Walk-forward (v8 priors)",
     lambda m: num(backtest.get((m["round"], m["match"]), {}).get("past_only_p_left")), "Structural"),
    ("loo", "Leave-one-out CV",
     lambda m: num(backtest.get((m["round"], m["match"]), {}).get("loo_p_left")), "Structural + parameter"),
    ("crowd", "Crowd",
     lambda m: None if num(m["left_support_pct"]) is None else num(m["left_support_pct"]) / 100, "None"),
]


def score(get, rows=None):
    """(correct calls, mean Brier score) of predictor `get` over `rows` (default: scored matches)."""
    rows = scored if rows is None else rows
    right, briers = 0, []
    for m in rows:
        p = get(m)
        y = 1.0 if m["winner"] == "L" else 0.0
        right += (p > 0.5) == (y == 1.0)
        briers.append((p - y) ** 2)
    return right, sum(briers) / len(briers)


SCORES = {k: score(get) for k, _, get, _ in PREDICTORS}
