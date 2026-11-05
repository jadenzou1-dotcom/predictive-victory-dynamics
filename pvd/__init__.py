"""Predictive Victory Dynamics: battle simulator, calibration, forecasting and bet sizing.

Data layout (all paths below are absolute, so modules work from any working directory):
  data/raw        hand-captured inputs: unit cards, placements, crowd snapshots, replay outcomes
  data/processed  derived from raw: frame-by-frame replay extracts, measured hits, posterior samples
  data/results    model outputs and the live match log: predictions, picks, scores, backtests
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = DATA / "results"
REPLAYS = RAW / "replays"                 # per-round outcomes read off each replay
EXTRACTED = PROCESSED / "replays"         # per-frame HP bars, curves, damage numbers
