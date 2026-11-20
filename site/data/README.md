# Website data (`site/data/`)

Every number on the results website comes from these JSON files. **Never type a number into the website by hand.** These files are generated, so do not edit them either. To update them after a result changes, run this from the repo root:

```
python scripts/export_site_data.py --check
```

The `--check` option prints the main table so you can compare it with `results/mg5/final/README.md`. For example, E1 greedy is 33.9 / 92.1 / 34.8 (uniform / peak / asym) and fixed-time is 140.5 / 172.3 / 159.8. The output is identical on every run: the t-SNE uses a fixed random seed.

Show delays with **1 decimal** (`x.toFixed(1)`), percentages with 0 decimals, and p-values as `p = 0.002` (or `p < 0.001`).

## Shared vocabulary

| Field | Values | Meaning |
|---|---|---|
| `experiment` | `normal` | Normal traffic, 3 scenarios × 10 seeds (RQ1, RQ3) |
| | `surge` | 3× demand from 1200 to 2400 s, never seen in training (RQ4) |
| | `sensor_failure` | All 4 queue detectors of light B1 read 0 (RQ4) |
| | `transfer` | E1 trained on 2x2, run on 4x4 without retraining (RQ5) |
| | `grid_2x2` | 2x2 grid results |
| `variant` | `standard` | Everything works |
| | `failure` | B1's detectors broken |
| | `failure_failsafe` | B1's detectors broken, B1 falls back to the tuned fixed-time cycle |
| | `failure_ablated` | B1's detectors broken and E1's messages replaced by their average (present only once those runs exist) |
| `controller` | `E1` `B1` `B2` `fixed` `tuned` `random` `E1_from2x2` | Display names are in `meta.json` → `controllers` |
| `mode` | `greedy` | The agent always takes its most likely action (deterministic) |
| | `sampled` | The agent samples from its policy (`--explore`) |
| | `none` | Controllers without a learned policy (fixed, tuned, random): they are the same in both modes, so show them next to both |
| `train_seed` | `0` `1` `2` | Training seed. Seed 0 is the main model and every result uses it. Seeds 1 and 2 exist only for E1 and B1 in normal traffic (the 3-seed check) |
| `scenario` | `uniform` `peak` `asym` `surge` | Traffic pattern |
| `seed` | `0` to `9` | Traffic (route file) seed. The same seed means the same cars for every controller |

Units: `true_delay` is seconds per vehicle (lower is better) and is the main metric. `teleports` counts vehicles SUMO removed after 300 s stuck; more than 0 means a gridlock happened.

## Files

### `summary.json`: one row per condition (use this for tables and bar charts)
A list of objects:
```
{"experiment":"normal","variant":"standard","grid":"4x4","controller":"E1","mode":"greedy","train_seed":0,
 "scenario":"uniform","n":10,"true_delay_mean":33.871,"true_delay_sd":1.35,"true_delay_ci95":[32.9,34.84],
 "teleports_mean":0.0,"entered_pct":99.98,"travel_time_mean":138.45,"switches_per_light_h_mean":331.8}
```
`switches_per_light_h_mean` is `null` for the fixed and tuned rows, because Hanin's file has no switch count.

To get the main table, filter `experiment=="normal" && grid=="4x4" && train_seed==0`.

### `episodes.json`: every single episode (use this for dot/strip plots and seed-by-seed views)
This file is compact: `{"columns":[...], "rows":[[...], ...]}`. Turn each row into an object with
`Object.fromEntries(columns.map((c, i) => [c, row[i]]))`. It has 960 episodes now, and 1,020 once the two ablation runs (30 episodes each) are added.

### `comparisons.json`: E1 against each other controller (use this for "E1 is X% lower, p = …")
One object per (experiment, variant, mode, scenario, other controller). The comparison is paired by traffic seed and uses training seed 0:
```
{"experiment":"normal","variant":"standard","grid":"4x4","mode":"greedy","scenario":"peak","other":"B1","n":10,
 "e1_mean":92.123,"other_mean":685.239,"diff_pct":-86.6,"e1_better_seeds":10,"wilcoxon_p":0.001953125,"ttest_p":8.3e-13}
```
A negative `diff_pct` means E1 is better. With 10 seeds the smallest possible Wilcoxon p is 0.002, so `p = 0.002` with `e1_better_seeds = 10` means E1 won every seed. Call a result "significant" only if `wilcoxon_p < 0.05`.

Once the message-ablation runs exist, there are also rows with `variant: "failure"` and `other: "E1_ablated"`. They compare E1 under sensor failure with its real messages (`e1_mean`) against the same E1 with its messages replaced by their average (`other_mean`). A negative `diff_pct` there means the real messages help.

### `training.json`: learning curves
One object per training run (`4x4_E1`, `4x4_E1_seed1`, …, `2x2_E1`):
`steps[]` and `return_per_agent_smoothed[]` (same length: every 5th iteration, a 20-iteration moving average, reward per traffic light) are the curve to plot. The other fields are `iterations`, `env_steps`, `train_hours`, `resumed_at_steps` (where Colab cut the run and it continued from a checkpoint) and `missing_iterations`. `4x4_B1_seed2` has 18 missing log rows (a logging gap only; the training was complete).

### `messages.json`: what E1's messages mean (RQ2)
- `tsne.points[]`: 2,500 messages (a fixed random sample of 103,680), each projected to 2-D. Each point is `{x, y, green, queue:[N,E,S,W], queue_total, msg_pc1}`. Colour by `green` (the sender's current green direction) or by `queue_total`. The colours should show that messages group by the sender's phase.
- `correlation`: `rows` (8 message dimensions plus PC1 and PC2) × `cols` (queues, green direction, time in phase), with `r[i][j]` the Pearson correlation. Show it as a heatmap with a diverging scale from -1 to 1.
- `ablation_normal_traffic`: E1 with its real messages vs with messages replaced by their average, in normal traffic (9 episodes). It gives `mean_normal`, `mean_ablated`, `worse_count` and `wilcoxon_p`.
- `report`: decoding scores, for example `decode_phase_accuracy` vs `phase_majority_baseline`, and `decode_queue_total_r2`.

### `meta.json`
`generated` (time), `git_commit`, `controllers` (display names), `units`, `sources` (every CSV used, with a short checksum) and `missing_files` (expected files that do not exist yet). The site footer should show "Data generated <generated> from commit <git_commit>".
