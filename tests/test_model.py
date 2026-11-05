"""Forecast blending, the published track record, and the simulator's basic contract."""
import random

import pytest

from pvd import blend, evaluate, sim


def test_blend_endpoints():
    # w = 1 is the model alone, w = 0 is the crowd alone (both clipped to [3%, 97%])
    assert blend.blend(0.8, 0.3, 1.0) == pytest.approx(0.8)
    assert blend.blend(0.8, 0.3, 0.0) == pytest.approx(0.3)
    assert blend.blend(0.99, 0.5, 1.0) == pytest.approx(0.97)


def test_blend_posterior_is_a_distribution():
    rows = blend.training_rows(blend.load())
    post = blend.posterior(rows)
    assert len(post) == len(blend.GRID)
    assert sum(post) == pytest.approx(1.0)
    assert min(post) >= 0


def test_track_record_matches_published_numbers():
    # regression test for the README scoreboard: (correct of 14, Brier)
    expected = {"live": (11, 0.234), "swf": (9, 0.210), "wf": (13, 0.106), "loo": (11, 0.114), "crowd": (6, 0.336)}
    assert len(evaluate.scored) == 14
    for k, (right, brier) in expected.items():
        r, b = evaluate.SCORES[k]
        assert r == right
        assert b == pytest.approx(brier, abs=5e-4)


def test_brier_of_a_coin_flip_is_a_quarter():
    rows = [dict(winner="L"), dict(winner="R")]
    assert evaluate.score(lambda m: 0.5, rows)[1] == pytest.approx(0.25)


@pytest.fixture(scope="module")
def final_match():
    units, places, _ = sim.load()
    return units, sim.match_units(places, "Final", "2")


def test_sim_returns_a_winner(final_match):
    units, plist = final_match
    random.seed(0)
    ids = {p["unit_id"] for p in plist}
    for _ in range(20):
        G, U = sim.draw_params(ids, [], {})
        winner, hp, t = sim.run(units, plist, G, U)
        assert winner in ("L", "R")


def test_sim_is_reproducible_with_a_seed(final_match):
    units, plist = final_match
    ids = sorted({p["unit_id"] for p in plist})

    def batch():
        random.seed(42)
        out = []
        for _ in range(10):
            G, U = sim.draw_params(ids, [], {})
            out.append(sim.run(units, plist, G, U)[0])
        return out

    assert batch() == batch()
