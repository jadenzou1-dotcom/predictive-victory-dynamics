# Archive

Early prototypes from the first rounds of the event, kept so the project's history is complete. Each was replaced by the `pvd/` package; they read the data layout from before the reorganization, so they aren't expected to run as-is.

| File | What it was | Replaced by |
|---|---|---|
| `allocation_curve.py` | Swept Strategy A's split between two bets and plotted Sharpe / EV / SD (`charts/R16-A_allocation.png`, `charts/R16-B_allocation.png`) | `pvd/strategies.py` (max certainty equivalent) |
| `calibrate.py` | First calibration: nested ABC rejection on replay outcomes (winner, remaining HP, wipe time) | `pvd/calibrate.py` (kernel ABC on full HP trajectories) |
| `replay_hp_tracker.py` | First OpenCV HP-bar tracker, team HP sampled 5×/s | `pvd/replay/extract.py` |

The round-by-round notes from the same period are in [`docs/rounds/`](../docs/rounds/).
