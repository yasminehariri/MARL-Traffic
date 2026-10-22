"""
Minimum-green bookkeeping shared by the scripts.

The environment reads MARL_MIN_GREEN once, when env.traci_env is first imported, so every
script must call set_min_green() BEFORE importing anything from env/.

A trained checkpoint must always be evaluated with the minimum green it was trained with.
train_4x4.py writes it to <run folder>/run_info.json; runs from before this option have no
"min_green" key and were trained without a minimum (0).
"""

import json
import os
import sys


def run_info_for(ckpt):
    """<run>/checkpoints/ckpt_000977 -> <run>/run_info.json (None if it does not exist)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(ckpt))), "run_info.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def min_green_for(ckpt, requested=None):
    """Minimum green to use with this checkpoint; exits if --min-green contradicts the training run."""
    info = run_info_for(ckpt)
    if info is None:
        if requested is None:
            sys.exit(f"No run_info.json next to {ckpt}: pass --min-green explicitly "
                     f"(0 for runs trained before the minimum-green rule).")
        return requested
    trained = int(info.get("min_green", 0))
    if requested is not None and requested != trained:
        sys.exit(f"{ckpt} was trained with min_green={trained} but --min-green {requested} was given. "
                 f"A policy must be evaluated under the rules it was trained with.")
    return trained


def set_min_green(value):
    if "env.traci_env" in sys.modules:
        sys.exit("internal error: set_min_green() called after env.traci_env was imported")
    os.environ["MARL_MIN_GREEN"] = str(int(value))
    print(f"Minimum green: {int(value)} s" + ("  (version 1 rules: no minimum)" if int(value) == 0 else ""),
          flush=True)
