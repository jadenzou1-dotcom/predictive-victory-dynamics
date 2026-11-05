"""Payout math and allocation: EV/variance of a bet, why Sharpe was dropped, and the CE search."""
import math

import pytest

from pvd.strategies import FLOOR, best_ce, leg, sharpe


def side(p, M, team="x"):
    return dict(team=team, p=p, M=M, sup=50)


def game(match, p_left, m_left, m_right):
    return dict(match=match, pm=p_left, pc=0.5,
                sides=[side(p_left, m_left, f"L{match}"), side(1 - p_left, m_right, f"R{match}")])


def test_leg_matches_binary_payoff():
    # e emblems pay e*M with prob p and e*FLOOR otherwise
    p, M, e = 0.6, 50, 7
    ev, var = leg(side(p, M), e)
    outcomes = [(p, e * M), (1 - p, e * FLOOR)]
    mean = sum(q * x for q, x in outcomes)
    assert ev == pytest.approx(mean)
    assert var == pytest.approx(sum(q * (x - mean) ** 2 for q, x in outcomes))


@pytest.mark.parametrize("p", [0.2, 0.5, 0.6, 0.785])
@pytest.mark.parametrize("M", [20, 50, 105, 200])
def test_sharpe_ignores_payout(p, M):
    # with a fixed floor, Sharpe of a binary bet is sqrt(p / (1 - p)) whatever the multiplier
    assert sharpe(side(p, M), 1) == pytest.approx(math.sqrt(p / (1 - p)))


def test_sharpe_cannot_see_a_strictly_worse_bet():
    # docs/DECISIONS.md #1: cutting the win payout from 50 to 20 halves EV, Sharpe is unchanged
    good, worse = side(0.6, 50), side(0.6, 20)
    assert leg(good, 1)[0] == pytest.approx(34.0)
    assert leg(worse, 1)[0] == pytest.approx(16.0)
    assert sharpe(good, 1) == pytest.approx(sharpe(worse, 1)) == pytest.approx(1.2247, abs=1e-4)


def test_tangency_weights_reproduce_r16b_split():
    # max-Sharpe weights of independent bets are proportional to excess EV / variance.
    # R16-B: Dye (78.5%, x105) has the higher Sharpe and EV, yet gets only 28% of the stake.
    dye, blaze = side(0.785, 105), side(0.605, 30)
    w = {}
    for name, s in (("dye", dye), ("blaze", blaze)):
        ev, var = leg(s, 1)
        w[name] = (ev - FLOOR) / var
    assert sharpe(dye, 1) > sharpe(blaze, 1)
    assert leg(dye, 1)[0] > leg(blaze, 1)[0]
    assert w["dye"] / (w["dye"] + w["blaze"]) == pytest.approx(0.28, abs=0.005)


def test_max_ev_spends_the_pool_on_the_better_side():
    games = [game("1", 0.7, 30, 105), game("2", 0.4, 50, 40)]
    ce, ev, var, picks = best_ce(games, 0.0, 200)
    # lambda = 0 is pure EV: every emblem earns >= FLOOR, so the whole pool goes out
    assert sum(e for e, _, _ in picks) == 200
    ev_per = {s["team"]: leg(s, 1)[0] for g in games for s in g["sides"]}
    for e, team, _ in picks:
        if e:
            m = team[1:]
            other = ("R" if team[0] == "L" else "L") + m
            assert ev_per[team] >= ev_per[other]
    assert ce == pytest.approx(ev)


def test_risk_aversion_trades_ev_for_lower_variance():
    games = [game("1", 0.3, 25, 105), game("2", 0.75, 30, 50)]
    _, ev0, var0, _ = best_ce(games, 0.0, 200)
    _, ev1, var1, _ = best_ce(games, 5e-4, 200)
    assert ev1 <= ev0 + 1e-9
    assert var1 < var0
