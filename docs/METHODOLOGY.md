# Methodology and results

The full write-up: the betting problem, how the battle model works and was calibrated, how it changed from v1 to v8, how it was evaluated without look-ahead bias, and what the money did. Reasoning behind each major choice is in [DECISIONS.md](DECISIONS.md); the dated, unedited log is in [JOURNAL.md](JOURNAL.md); every match is in [../RESULTS.md](../RESULTS.md).

**Contents**
1. [The problem: a parimutuel binary bet](#1-the-problem-a-parimutuel-binary-bet)
2. [The battle model and what was measured](#2-the-battle-model-and-what-was-measured)
3. [Model versions v1 to v8](#3-model-versions-v1-to-v8)
4. [Evaluation and look-ahead bias](#4-evaluation-and-look-ahead-bias)
5. [What worked, and what went wrong](#5-what-worked-and-what-went-wrong)
6. [Betting results](#6-betting-results)

## 1. The problem: a parimutuel binary bet

Each match is an auto-battle between two unit teams. Players stake Victory Emblems on a side before the support rate locks. A right pick pays `emblems × multiplier`; a wrong pick still pays `emblems × 10`. The less popular side pays the higher multiplier, so the crowd's support rate works like parimutuel odds: it is an implied probability, and the multiplier is the price.

The edge therefore comes from two places: a better win probability than the crowd's, and the payout asymmetry when a strong side is priced as a long shot.

### Event rules

| Rule | Value | Source |
|---|---|---|
| Emblems | 100 granted per day, hold cap 300 | in-game counter |
| Stake per match | up to 300 | slider max (Jaden) |
| Wrong pick | pays 10 per emblem | scoring rules |
| Right pick | pays emblems × multiplier | scoring rules |
| Support lock | ~4 h before the round ends (~18:00 ET); round ends ~22:00 ET | event screen |
| Final | 2 matches: 3rd-place playoff and Championship | event screen |

### Payout math

`EV per emblem = p · M + (1 − p) · 10`

The multiplier `M` is a **step function** of support rate, not a smooth curve. Observed pairs:

| Support | Multiplier | Source |
|---|---|---|
| 13–14% | ×200 | own screens (R16-A, Final) |
| 15–25% | ×105 | own screens (R16-B, QF-A, QF-B, Final) + Game8 |
| 27–35% | ×70 | own (QF-A 35%) + Game8 |
| 39% | ×70 (ours) or ×50 (Game8) | conflicting edge |
| 42–47% | ×50 | own (SF) |
| 53–58% | ×40 | own (SF) |
| 61% | ×35 (ours) or ×40 (Game8) | conflicting edge |
| 65–70% | ×35 | own (QF-A) + Game8 |
| 73–83% | ×30 | own + Game8 |
| 86–88% | ×25 | own (R16-A, Final) + Game8 |

Tier edges are fuzzy, since displayed support is rounded. Always read the multiplier on screen at the lock.

Break-even win probability for side A against side B is `(M_B − 10) / ((M_A − 10) + (M_B − 10))`. `pvd/strategies.py` prints it for every match.

## 2. The battle model and what was measured

`pvd/sim.py` is an agent-based Monte Carlo simulation: units on a grid with stats from their cards, targeting, movement, kiting, AoE splash, crowd control, healing, elemental auras and reactions, and overtime. Every mechanic that can't be read off a card is a parameter with a prior. `pvd/calibrate.py` fits those parameters to the replays with kernel approximate Bayesian computation (ABC): it compares per-second simulated HP curves against the curves `pvd/replay/` extracts from the replay videos (OpenCV HP-bar segmentation and OCR of the clock, commentary and damage numbers). Each prediction averages over posterior draws, so the output probability carries the model's uncertainty.

Everything below is treated as known. Values marked *fitted* come from calibration; everything else was measured directly from frames, damage numbers or the event screens.

### Battle rules

| Mechanic | Value | Evidence |
|---|---|---|
| Battle clock | 45 s | every replay |
| Damage per hit | 0.449 × card ATK × skill multiplier (50% / 100% / 150% / 200%) | damage numbers; 0.449 = Genshin DEF × RES at level 80 (0.5 × 0.9) |
| Counter triangle | ~1.0–1.08× on damage | frame checks (R16-A/B/C) |
| Transformative reactions | flat ~800 (Electro-Charged, Superconduct, Overloaded) | recurring 795/800 numbers across very different attackers |
| Squad HP | every member has full card HP | damage-number totals vs HP-bar loss (R16-B, R16-C) |
| Targeting | *fitted*: lowest HP 0.49, nearest 0.34, sticky random 0.17 (v8) | trajectory fit; not settled |
| Overtime | one hit of ~11% max HP to every unit ~1.3–1.5 s after 00:00; repeats every ~8 s (*fitted*: 5–11 s) | step drops in QF-A M1, QF-B M1, SF M2; no 2nd pulse within 4.4 s |
| Freeze | Hydro + Cryo; *fitted* 1–5.5 s; bosses most likely unfreezable | Lawachurl frozen 10 s straight (R16-A) |

### Unit specifics

| Unit | Finding | Evidence |
|---|---|---|
| Sappho (boss) | volleys splash 3 targets at full damage (8,737 = 0.45 × ATK); self-freezes enemies; kites melee | R16-A frame study, commentary |
| Sappho frost field | opening hit 5,461 (0.281 × ATK) on all inside, then 1,092 Cryo ticks (0.0563 × ATK); first cast ~10–12 s, radius ~2 cells (*fitted*) | SF M1 frames 00:35–00:32; commentary "especially deadly against fragile ranged units" |
| Swiftshadow Ninken | normal hit 4,854 = 0.449 × ATK; post-kill plunge lands as the same 4,854, AoE on several targets | QF-B frames 69.9–70.3 s |
| Slimes | carry their own element permanently; Electro hits on them show 795 reaction numbers | QF-B frames |
| Large Cryo Slime | hit 578 = 0.148 × ATK | R16-D / QF-B |
| Treasure Hoarder Scout | hit 1,033 = 0.449 × ATK | R16-B (11 hits) |

Per-hit measurements live in `data/processed/unit_hits_measured.csv`; the full unit cards in `data/raw/units.csv`.

## 3. Model versions v1 to v8

**How fit quality is measured:** calibration is kernel ABC on per-second replay trajectories (`pvd/calibrate.py`). A smaller tolerance (EPS) at the same effective sample size (ESS) means the model reproduces the replays more closely. Fit numbers are only comparable at the same replay count.

| Version | Date | Key change | Replays fit | Fit | Used for |
|---|---|---|---|---|---|
| v1 | Oct 12 | Deterministic stat-pool battle | 0 | — | Rejected (0% / 99.9%) |
| v2 | Oct 12 | Monte Carlo over unknown mechanics | 0 | — | R16-A |
| v3 | Oct 13–16 | Elements and reactions, kiting, 45 s clock, replay calibration, measured hit size | 2–6 | EPS 0.10, ESS 70 | R16-B, R16-C |
| v4 | Oct 16 | Hit = 0.45 × ATK (DEF × RES); Sappho splash; counters ≈ 1×; long freezes | 6 | EPS 0.10, ESS 42 | — |
| v5 | Oct 16–18 | Panel round 1 fixes | 6 → 10 | ESS 68 (6); EPS 0.15, ESS 124 (10) | QF-A, QF-B |
| v6 | Oct 18 | Squad HP fixed from data; Ninken plunge; slime self-aura; measured hit priors | 10 | EPS 0.12, ESS 72 | **SF bets** |
| v7 | Oct 18 | Overtime = one-off pulse; targeting fitted; plunge = 1× | 10 → 12 | EPS 0.12, ESS 106 (10); EPS 0.15, ESS 138 (12) | Final (first pass) |
| v8 | Oct 19 | Sappho frost field | 12 | **EPS 0.15, ESS 188** | **Final bets** |

### v1: deterministic sim (rejected)

Listed ATK was treated as damage per hit, and the bigger stat pool always won (0% / 99.9%). Listed ATK isn't DPS: skill multipliers and attack speed are hidden.
### v2: Monte Carlo over the unknowns

Each run samples the unknowns:
- counter strength;
- squad HP rule (per member vs split);
- damage multiplier and attack interval per unit;
- splash;
- crowd control;
- healing.

Lesson from R16-A: counter tags are a tie-breaker; raw HP × ATK dominates.
### v3: after the first replays

- **R16-A replay findings:**
  - a fixed 45 s clock with no drain inside it;
  - Sappho freezes enemies by herself (Hydro + Cryo);
  - ranged bosses kite melee.
- **Added:** elemental auras and reactions (Freeze, Vaporize/Melt, Electro-Charged, Superconduct, Swirl), dual-element units, a shatter bonus, kiting.
- **Calibration:** first a 3-number replay summary (an early calibrator, since removed), then full per-second trajectories (`pvd/calibrate.py`) from the replay extractor (`pvd/replay/`).
- **Damage-number reader (`pvd/replay/dmg.py`):** transformative reactions deal a flat ~800; a normal hit is a fixed fraction of ATK.
### v4: hit-size decomposition

- **Damage per hit:** 0.449 × ATK is exactly Genshin's DEF × RES factor at level 80 (0.5 × 0.9). Other hit sizes are skill multipliers (50% / 150% / 200%).
- **Sappho frame study:**
  - counters don't change damage;
  - her attacks splash at full damage;
  - freezes can last 10 s;
  - bosses may be unfreezable (new `boss_freeze` parameter).
### v5: panel round 1 fixes

- **Sappho and bosses:**
  - Sappho's splash hits the 3 nearest enemies;
  - bosses ignore immobilize.
- **Reactions:**
  - Anemo/Pyro hits end a Freeze;
  - Swirl spreads the absorbed element.
- **Calibration:**
  - shrinkage per replay group;
  - unit bar counts only from bars ≥ 30 px;
  - squad alive-share down-weighted.

Results:
- QF-A: both underdogs the model favoured won.
- QF-B: the model called both favourites correctly.
### v6: implementing the SF panel's fixes (Jaden: "just fix them")

1. **Squad HP rule settled from data:** damage-number totals vs HP-bar loss in R16-B (Fruity) and R16-C (Chubby). Only "full card HP per member" lands in the reader's capture range (0.43 / 0.36 vs 0.23–0.43 on single-unit teams); shared HP gives 0.82 / 0.95. `per_member` is now fixed to 1.
2. **Ninken post-kill plunge** added.
3. **Slimes carry their own element permanently,** so reactions fire on them.
4. **Unit damage priors** centred on measured per-hit damage (`data/processed/unit_hits_measured.csv`).

SF: P(Fruity) 0.91, P(Grenadier) 0.80. Grenadier won; Fruity lost (see v8).
### v7: overtime, targeting, plunge

- **Overtime is a pulse, not a drain.** Every overtime replay shows a single ~11% max-HP hit ~1.3–1.5 s after 00:00, then flat bars for 2.5–4.4 s. Modelled as pulses every `ot_period` s (fit ≈ 8 s).
- **Targeting is now a fitted choice** (nearest / lowest HP / sticky random). The replays lean lowest HP (0.49–0.61), but it isn't settled.
- **Ninken plunge = a normal hit (1×) as AoE,** read off the QF-B frames. The earlier "3×" fit was compensating for nearest-enemy targeting.
### v8: Sappho's frost field (the SF M1 miss)

SF M1 frames show a large field dropped on the enemy cluster:
- an opening hit of 5,461 (0.281 × ATK, Hydro) on everyone inside;
- then repeated 1,092 Cryo ticks (0.0563 × ATK).

That's ~12k per unit per field: it nearly one-shots 13k Scouts and freezes the whole cluster. Damage is measured from the frames; timing, radius and lifetime are fitted. First cast fits at 11.6 s (frames: ~10 s).

Results:
- Best fit so far.
- In-sample, SF M1 now favours Sappho (0.67), as happened.
- Final: P(Fruity) 0.91, P(Saurians) 0.87.

## 4. Evaluation and look-ahead bias

The live model is scored against three backtests of the final model (v8) and against the crowd. The rows differ in how much look-ahead bias they let in:
- **Live:** the calls actually made before each match, by whichever model version existed that day. No look-ahead bias at all.
- **Strict walk-forward:** v8's battle engine, rebuilt for each round from only what was knowable before it (`pvd/backtest/walk_forward_strict.py`):
  - **Measured values:** every value measured from a replay (overtime pulse size, hit sizes, squad HP rule, flat reaction damage, counters ≈ 1×) gets a vague prior until the round after that replay.
  - **Discovered mechanics:** every mechanic discovered in a replay (frost field, Ninken plunge, slime self-aura, Sappho's splash) is switched off until then.
  - **Fit:** the model is then fitted to earlier replays only.
  - **Remaining leak:** the general engine design (grid, targeting modes, reactions) was built up during the event.
- **Walk-forward (v8 priors):** v8 refit on earlier rounds only, but starting from v8's priors and mechanics. Those were measured in later replays: an early match "knows" overtime costs ~11% max HP before any overtime had been seen. That is structural leakage.
- **Leave-one-out cross-validation:** v8 refit on every other match. Later rounds also leak into the parameters.

<!-- track-record:start -->
<!-- generated by scripts/make_results.py from data/results/matches.csv, backtest_v8.csv and backtest_strict_wf.csv; do not edit by hand -->

**Scoreboard: the 14 matches with a live model call**

| Predictor | How the prediction was made | Look-ahead bias | Correct | Brier ↓ |
|---|---|---|---|---|
| **Live model** | The call actually made before the match, by the model version of the day (v1–v8) | **None**: true out-of-sample | **11 / 14** | 0.234 |
| Strict walk-forward | v8's battle engine, but each measured value or replay-discovered mechanic is held back until the round after its source replay; refit on earlier rounds only (`pvd/backtest/walk_forward_strict.py`) | Minimal: only the engine's general design (grid, targeting, reactions) came later | 9 / 14 | 0.210 |
| Walk-forward (v8 priors) | Final model (v8) with its full priors, refit on earlier rounds only | Structural: its priors and mechanics (overtime pulse, frost field, hit sizes, squad HP) were measured in later replays | 13 / 14 | 0.106 |
| Leave-one-out CV | Final model (v8), refit on every other match, earlier and later | Structural + parameter: later rounds are also in the fit | 11 / 14 | 0.114 |
| Crowd | Support rate read as a win probability | None | 6 / 14 | 0.336 |

- **Live is the headline.** It is the only row with no look-ahead bias at all.
- **Strict walk-forward (9/14) is the fair backtest.** It replays the final engine with only what was knowable before each match. It lands near live (11/14): it misses all three of live's misses (R16-A M2, R16-B M1, SF M1) plus R16-B M2 (a 52% call) and QF-B M1. Its Brier score is a little better than live's, so its probabilities are better calibrated, but it picks fewer winners.
- **Walk-forward with v8 priors (13/14) is flattered by look-ahead.** Its only difference from strict is priors and mechanics measured in later replays, so the jump from 9 to 13 is mostly leakage. Two of its wins (R16-B) are 50–51% coin flips anyway.
- **Noise:** one row beating another by 1–2 matches is within noise. On the Final, where strict and v8 have the same information, two separate fits give 0.92 / 0.74 and 0.92 / 0.88.
- **Brier score** is the mean squared error of the probability (0 is perfect, 0.25 is always saying 50%). It rewards calibrated confidence, not just the right side.
- **Why 14, not 16:** the sim needs every unit's stats and grid position. For R16-D the losing teams' unit cards and the board were never captured, so no model (live or backtest) can run those two matches. The crowd went 8 / 16 counting them.

**Every match**

Each model or crowd cell shows the side it favoured (L/R) and how confident it was. The winner is in bold.

| Round | Left vs Right | Live model | Strict walk-forward | Walk-forward (v8 priors) | Leave-one-out | Crowd | Winner |
|---|---|---|---|---|---|---|---|
| R16&#8209;A&nbsp;M1 | Howl Crew vs **Fungi Clan** | R&nbsp;92%&nbsp;✅ | R&nbsp;97%&nbsp;✅ | R&nbsp;97%&nbsp;✅ | R&nbsp;98%&nbsp;✅ | R&nbsp;75%&nbsp;✅ | R |
| R16&#8209;A&nbsp;M2 | Frostarm & Shaman vs **Saurians** | L&nbsp;88%&nbsp;❌ | L&nbsp;80%&nbsp;❌ | R&nbsp;88%&nbsp;✅ | R&nbsp;83%&nbsp;✅ | L&nbsp;87%&nbsp;❌ | R |
| R16&#8209;B&nbsp;M1 | Dye Experts vs **Mirror Fruity** | L&nbsp;78%&nbsp;❌ | L&nbsp;59%&nbsp;❌ | R&nbsp;51.3%&nbsp;✅ | L&nbsp;52.0%&nbsp;❌ | R&nbsp;79%&nbsp;✅ | R |
| R16&#8209;B&nbsp;M2 | **Blazing Beasts** vs Rude-a-Lot | L&nbsp;60%&nbsp;✅ | R&nbsp;52.5%&nbsp;❌ | L&nbsp;50.2%&nbsp;✅ | R&nbsp;54.0%&nbsp;❌ | L&nbsp;78%&nbsp;✅ | L |
| R16&#8209;C&nbsp;M1 | **Grenadier** vs Seahorse & Crab | L&nbsp;62%&nbsp;✅ | L&nbsp;56%&nbsp;✅ | L&nbsp;53.2%&nbsp;✅ | L&nbsp;56%&nbsp;✅ | L&nbsp;85%&nbsp;✅ | L |
| R16&#8209;C&nbsp;M2 | **Chubby Crisis** vs Consecrated Beast | L&nbsp;51.0%&nbsp;✅ | L&nbsp;90%&nbsp;✅ | L&nbsp;85%&nbsp;✅ | L&nbsp;88%&nbsp;✅ | R&nbsp;77%&nbsp;❌ | L |
| R16&#8209;D&nbsp;M1 | **Ninken** vs Hilichurl Fighters | — | — | — | — | L&nbsp;82%&nbsp;✅ | L |
| R16&#8209;D&nbsp;M2 | Ultra-Ancient vs **Round-and-Frost** | — | — | — | — | R&nbsp;66%&nbsp;✅ | R |
| QF&#8209;A&nbsp;M1 | **Saurians** vs Fungi Clan | L&nbsp;81%&nbsp;✅ | L&nbsp;92%&nbsp;✅ | L&nbsp;90%&nbsp;✅ | L&nbsp;91%&nbsp;✅ | R&nbsp;65%&nbsp;❌ | L |
| QF&#8209;A&nbsp;M2 | **Mirror Fruity** vs Blazing Beasts | L&nbsp;77%&nbsp;✅ | L&nbsp;84%&nbsp;✅ | L&nbsp;84%&nbsp;✅ | L&nbsp;80%&nbsp;✅ | R&nbsp;85%&nbsp;❌ | L |
| QF&#8209;B&nbsp;M1 | **Grenadier** vs Chubby Crisis | L&nbsp;61%&nbsp;✅ | R&nbsp;66%&nbsp;❌ | L&nbsp;56%&nbsp;✅ | L&nbsp;54.2%&nbsp;✅ | L&nbsp;77%&nbsp;✅ | L |
| QF&#8209;B&nbsp;M2 | **Ninken** vs Round-and-Frost | L&nbsp;56%&nbsp;✅ | L&nbsp;62%&nbsp;✅ | L&nbsp;68%&nbsp;✅ | L&nbsp;68%&nbsp;✅ | L&nbsp;77%&nbsp;✅ | L |
| SF&nbsp;M1 | Mirror Fruity vs **Saurians** | L&nbsp;91%&nbsp;❌ | L&nbsp;86%&nbsp;❌ | L&nbsp;59%&nbsp;❌ | L&nbsp;61%&nbsp;❌ | L&nbsp;58%&nbsp;❌ | R |
| SF&nbsp;M2 | **Grenadier** vs Ninken | L&nbsp;80%&nbsp;✅ | L&nbsp;83%&nbsp;✅ | L&nbsp;80%&nbsp;✅ | L&nbsp;79%&nbsp;✅ | R&nbsp;53.0%&nbsp;❌ | L |
| 3rd&nbsp;place | **Mirror Fruity** vs Ninken | L&nbsp;91%&nbsp;✅ | L&nbsp;92%&nbsp;✅ | L&nbsp;92%&nbsp;✅ | L&nbsp;92%&nbsp;✅ | R&nbsp;86%&nbsp;❌ | L |
| Final | **Saurians** vs Grenadier | L&nbsp;87%&nbsp;✅ | L&nbsp;74%&nbsp;✅ | L&nbsp;88%&nbsp;✅ | L&nbsp;88%&nbsp;✅ | R&nbsp;75%&nbsp;❌ | L |
| **Correct** | *out of the 14 matches with a model call* | **11&nbsp;/&nbsp;14** | **9&nbsp;/&nbsp;14** | **13&nbsp;/&nbsp;14** | **11&nbsp;/&nbsp;14** | **6&nbsp;/&nbsp;14** | 14&nbsp;/&nbsp;14 |

<!-- track-record:end -->

**Could it be luck?** If both model and crowd were coin flips:
- Over 14 matches, the model beats the crowd by 5 or more (the actual gap) about **4%** of the time.
- The model alone gets 11 or more of 14 right about **3%** of the time.

So the evidence points to a real edge, but 14 matches is a small sample.

**When the model and crowd disagreed,** the model was right **6 of 7** times (R16-C M2, QF-A M1, QF-A M2, SF M2 and both Final matches; the crowd won R16-B M1).

## 5. What worked, and what went wrong

### What worked

- **The Final:** the crowd backed Ninken (86%) and Grenadier (75%). The model gave Fruity 91% and Saurians 87%, and stress tests never took either below 62%. Both won at ×200 and ×105, worth +40,000 (Dino) and +27,365 (Shado) in one round.
- **Payout asymmetry:** the biggest wins came from strong sides the crowd under-rated. Two examples: Saurians in R16-A (Dino; ×200 at the last snapshot), and both QF-A underdogs (Saurians ×70, Fruity ×105).
- **Reading mechanics off the replays instead of guessing:**
  - hit size (DEF × RES);
  - flat reaction damage;
  - squad HP rule;
  - overtime pulse;
  - Ninken plunge;
  - Sappho frost field.
- **Fit quality:** each discovery tightened the fit to the replays; v8 is the best yet.
- **Catching a flawed objective:** Strategy A started as max-Sharpe. A worked counterexample showed that Sharpe ignores payout when stakes can't be levered, so A switched to certainty equivalent ([DECISIONS.md, section 1](DECISIONS.md#1-sharpe-ratio-was-the-wrong-objective-and-i-only-caught-it-by-intuition)).
- **Calibration check:** after QF-B, the real margins landed at the 59th–65th percentile of the model's own outcome distribution, i.e. ordinary.
- **Stress-testing before big bets:** Final picks hold under the worst-case changes tested:
  - Ninken ×2 damage;
  - wider AoE;
  - weaker frost field;
  - Sappho ×0.6.

### What went wrong, and what it taught

- **R16-A M2 and R16-B M1 (early misses):** v1–v3 lacked reactions, Sappho's self-freeze and real hit sizes, and the edge on unseen units was thin. → Model more mechanics, temper unseen lineups, save emblems for later rounds.
- **SF M1 (Fruity 0.91, lost):** Sappho's frost field was unknown. Even v8, which knows the field, misses this match unless it has seen this replay, so the field's strength was learned from this one fight. → Trust was cut from 0.6 to 0.5 by the panel; the backtest supports 0.65.
- **Process gaps:**
  - the damage-number reads for QF-A and SF never ran to completion;
  - unit positions and targeting are not tracked frame by frame;
  - R16-D couldn't be used.

## 6. Betting results

Two accounts ran one objective each, as a controlled comparison:
- **Shado (Strategy A):** maximise the certainty equivalent `EV − (λ/2)·Var` with λ = 2.1e-4 (risk-adjusted).
- **Dino (Strategy B):** maximise EV.

The two accounts finished **1st (Dino, 82,500) and 2nd (Shado, 46,450)** on the friends leaderboard; the best other friend finished on 16,600.

### Points by round

Every per-match score is in `data/results/matches.csv`. The running totals match the verified pre-QF-B totals and the final leaderboard exactly.

| Round | Shado (A) | Dino (B) | Shado total | Dino total |
|---|---|---|---|---|
| R16-A | +2,020 | +20,000 (Saurians ×200) | 2,020 | 20,000 |
| R16-B | +2,440 | +1,000 | 4,460 | 21,000 |
| R16-C, R16-D | no bets | no bets | 4,460 | 21,000 |
| QF-A | +8,225 | +10,500 | 12,685 | 31,500 |
| QF-B (both bets on underdogs that lost) | +1,000 | +1,000 | 13,685 | 32,500 |
| SF | +5,400 | +10,000 | 19,085 | 42,500 |
| **Final** | **+27,365** | **+40,000** | **46,450** | **82,500** |

### Final (locked Oct 19, 20:06 ET): both underdogs won

| Match | Model (v8) | Crowd | Pooled (w = 0.65) | Multiplier | Break-even |
|---|---|---|---|---|---|
| 3rd place: **Mirror Fruity** vs Ninken | 0.91 | 0.14 | 0.64 | ×200 / ×25 | 0.073 |
| Championship: **Saurians** vs Grenadier | 0.87 | 0.25 | 0.65 | ×105 / ×30 | 0.174 |

Worst stress tests: Fruity 0.62, Saurians 0.66.

**Bets:**
- Shado: 67 Fruity + 133 Saurians (EV ~18,400).
- Dino: 200 Fruity (EV ~26,400).

**Result:** Mirror Fruity beat Ninken and Saurians beat Grenadier, both confirmed in the replay and on the results screen. Shado scored 67 × 200 + 133 × 105 = 27,365 against an EV of ~18,400. Dino scored 200 × 200 = 40,000 against an EV of ~26,400.

*Account names:* Shado appears as "Shadow" in the older notes and the CSVs.
