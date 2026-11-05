"""Command-line entry point: python -m pvd <command> [args...]

Each command runs one pipeline stage as its own module, so multiprocessing workers in the
calibration and backtest stages import cleanly. PYTHONHASHSEED is pinned so seeded runs are
reproducible: the sim iterates over sets of unit ids, whose order otherwise changes per process.
"""
import os
import subprocess
import sys

COMMANDS = {
    "sim":             ("pvd.sim",                          "P(left wins), uncertainty and EV per emblem for a round"),
    "calibrate":       ("pvd.calibrate",                    "refit the sim to every replay (kernel ABC) -> data/processed/"),
    "blend":           ("pvd.blend",                        "fit the model/crowd log-odds weight; leave-one-out log loss"),
    "strategies":      ("pvd.strategies",                   "emblem allocation: max-CE (lambda), max-EV and benchmarks"),
    "backtest-loo":    ("pvd.backtest.loo",                 "leave-one-out and walk-forward (v8 priors) backtests"),
    "backtest-strict": ("pvd.backtest.walk_forward_strict", "strict walk-forward: only what was knowable before each round"),
    "margin-check":    ("pvd.backtest.margin_check",        "where real margins fall in the model's outcome distribution"),
}


def usage():
    print("usage: python -m pvd <command> [args...]\n\ncommands:")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:16s} {desc}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        usage()
        sys.exit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
    module = COMMANDS[sys.argv[1]][0]
    env = {**os.environ, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "0")}
    sys.exit(subprocess.call([sys.executable, "-m", module, *sys.argv[2:]], env=env))
