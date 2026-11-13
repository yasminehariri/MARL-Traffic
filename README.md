# Emergent Communication in Multi-Agent Traffic Signal Control

Final Year Project, Rafik Hariri University, 2026. Yasmine El Hariri and Hanin Khalil.

Sixteen traffic lights on a 4x4 SUMO grid are each controlled by a reinforcement-learning agent (PPO with a shared policy, RLlib). We compare four controllers:

| ID | Controller |
|---|---|
| B0 | Fixed-time (SUMO default program), plus a tuned fixed-time variant |
| B1 | MARL, no communication |
| B2 | MARL, hand-crafted communication (neighbours share their queue lengths) |
| E1 | MARL, learned communication (CommNet-style continuous messages) |

**Final results and their interpretation:** [`results/mg5/final/README.md`](results/mg5/final/README.md). It covers RQ1 to RQ5, all tables, the statistics and the figures.

## Repository layout

| Folder | Contents |
|---|---|
| `env/` | SUMO/TraCI environment (`traci_env.py`) and the RLlib wrappers for B1, B2, E1 |
| `configs/` | SUMO networks (2x2, 4x4) and route files (3 scenarios x 10 evaluation seeds, training seeds 100 to 104, surge) |
| `scripts/` | Route generation, training, evaluation, analysis, message interpretation, demo recording and video |
| `results/mg5/` | Final (version 2, 5 s minimum green) evaluation CSVs, training logs, message data |
| `results/mg5/final/` | Final tables, statistical tests and figures |
| `results/` (other) | Version 1 results (no minimum green), kept for the record |
| `demo/` | 3D replay of recorded simulations (see [`demo/README.md`](demo/README.md)) |
| `tests/` | Environment and model tests |

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
```

SUMO is installed by pip (`eclipse-sumo`). On Colab we used `ray.init(num_cpus=2, object_store_memory=500MB)` and `torch.set_num_threads(1)`; the scripts already do this.

## Reproduce

All commands run from the repo root. Paths below are the Colab paths we used; change `--out` and `--ckpt` for another machine.

**1. Routes** (already in `configs/`, so this is only needed to regenerate them):
```
python scripts/generate_routes_4x4.py --everything
python scripts/generate_routes_4x4.py --everything --grid 2x2
python scripts/generate_routes_4x4.py --surge
```

**2. Training** (500k environment steps, about 3 to 3.5 hours each on Colab; the 2x2 run takes about 1.2 hours):
```
python scripts/train_4x4.py --cond E1 --reward local
python scripts/train_4x4.py --cond B1 --reward local
python scripts/train_4x4.py --cond B2 --reward local
python scripts/train_4x4.py --cond E1 --reward local --grid 2x2
```
Checkpoints go to `MARL_checkpoints/<grid>/<cond>_local_mg5/`. Add `--resume` to continue an interrupted run.

**3. Evaluation** (seeds 0 to 9, scenarios uniform, peak and asym). One run per controller and mode, for example:
```
python scripts/evaluate_4x4.py --cond E1 --ckpt <E1 checkpoint> --out MARL_results/4x4_eval_mg5_E1.csv
python scripts/evaluate_4x4.py --cond E1 --ckpt <E1 checkpoint> --explore --label E1_explore --out MARL_results/4x4_eval_mg5_explore_E1.csv
python scripts/evaluate_4x4.py --cond E1 --ckpt <E1 checkpoint> --scenarios surge --out MARL_results/4x4_surge_mg5_E1.csv
python scripts/evaluate_4x4.py --cond fixed --scenarios surge --out MARL_results/4x4_surge_mg5_fixed.csv
python scripts/evaluate_4x4.py --cond E1 --ckpt <2x2 E1 checkpoint> --label E1_from2x2 --out MARL_results/4x4_eval_mg5_E1_from2x2.csv
python scripts/evaluate_4x4.py --cond E1 --ckpt <2x2 E1 checkpoint> --grid 2x2 --out MARL_results/2x2_eval_mg5_E1.csv
```
The same pattern applies for B1, B2, tuned and random. The minimum green is read from each checkpoint automatically. Greedy evaluation is deterministic, and sampled (`--explore`) evaluation is seeded per episode, so reruns give identical numbers.

**4. Message analysis (RQ2):**
```
python scripts/check_messages_4x4.py --ckpt <E1 checkpoint> --eval-csv MARL_results/4x4_eval_mg5_E1.csv
python scripts/interpret_messages_4x4.py --npz results/mg5/e1_messages_4x4_mg5/messages_4x4.npz --report results/mg5/e1_messages_4x4_mg5/message_report.json --out results/mg5/final/messages
```

**5. Tables, statistics and figures:** the exact commands are at the end of [`results/mg5/final/README.md`](results/mg5/final/README.md).

**6. Demo:** see [`demo/README.md`](demo/README.md).
