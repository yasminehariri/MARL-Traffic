# Final results, version 2 (5 s minimum green)

All numbers are mean **true delay per vehicle in seconds** (lower is better) over evaluation seeds 0 to 9. Every controller saw the same route files (pairing check passed). Statistics: paired t-test and exact Wilcoxon, Holm-corrected; with 10 seeds the smallest possible Wilcoxon p is 0.0293 (0.0098 for one scenario).

"Greedy" means trained agents take their most likely action. "Sampled" means they sample from their policy (seeded, so reproducible). Fixed, tuned and random are the same in both modes.

## Folder contents

| Folder | What it answers |
|---|---|
| `normal_greedy/`, `normal_sampled/` | RQ1 and RQ5: all controllers, 3 scenarios |
| `greedy_vs_sampled/` | How much each learned controller depends on the action mode |
| `messages/` | RQ2: t-SNE, correlation heatmap, temporal plot, ablation |
| `transfer_2x2_to_4x4/`, `grid_2x2/`, `training/` | RQ3: zero-shot transfer, 2x2 results, convergence curves |
| `surge_greedy/`, `surge_sampled/` | RQ4: 3x demand surge (never seen in training) |
| `rq5_all_metrics.csv` | RQ5: every controller on every metric (delay, travel time, throughput, teleports, switches) |

## Normal traffic

| Scenario | E1 greedy | E1 sampled | B1 greedy | B1 sampled | B2 greedy | B2 sampled | Tuned | Fixed | Random |
|---|---|---|---|---|---|---|---|---|---|
| Uniform | 33.9 | 36.1 | **29.5** | 32.0 | 47.0 | 45.3 | 93.2 | 140.5 | 122.1 |
| Peak | **92.1** | 95.3 | 685.2 | 172.8 | 854.9 | 426.0 | 169.5 | 172.3 | 323.7 |
| Asym | 34.8 | 37.5 | **26.6** | 34.6 | 207.8 | 78.1 | 126.2 | 159.8 | 251.1 |

- **E1 is the only learned controller that never gridlocks**: 0 teleports in all 60 normal runs, 100% of vehicles entered.
- **Peak hour, E1 is best of all controllers in both modes.** Greedy: 87% lower than B1 and 89% lower than B2. Sampled: 45% lower than B1 and 78% lower than B2. It is better than B1 on 10 of 10 seeds in every peak comparison, all significant.
- **Light traffic (uniform, asym), B1 is best.** E1 is 3 to 8 s slower than B1 (significant except sampled asym, p = 0.19).
- **E1 vs B2:** E1 is lower in every scenario and both modes. Significant at peak (both modes) and sampled asym; sampled uniform is significant by Wilcoxon only (t-test p = 0.10); greedy uniform and greedy asym are not significant because B2's results vary too much between seeds (B2 greedy asym sd = 212 s).
- **B2 (hand-crafted messages) is worse than B1 (no messages) in every scenario and both modes**, 10 of 10 seeds each, all significant.
- **Against fixed-time:** every learned controller beats it in uniform and asym, except B2 greedy in asym (208 vs 160 s). At peak only E1 beats it, in both modes (by 45 to 47%).

### Greedy vs sampled

E1 changes by only 3 to 8% between modes. B1 and B2 collapse when greedy at peak (B1 173 to 685 s, B2 426 to 855 s) because the deterministic policy locks into gridlock. E1's policy is the most stable of the three.

## RQ2: what the messages contain

- The strongest message dimension vs traffic variable correlation is |r| = 0.70 (green phase W); every one of the 8 dimensions has |r| > 0.5 with the sender's current green phase. **This meets the scope's minimum answer (r > 0.5).**
- The receiver can recover the sender's green phase from the message with 100% accuracy (majority baseline 39%), and the sender's total queue with R² = 0.57.
- The t-SNE map splits cleanly by green phase, and the most congested states form their own end-of-arm clusters.
- The messages are effectively **one-dimensional**: the first principal component explains 99.9% of the variance (participation ratio 1.0). E1 learned a compact "which phase am I in, how loaded am I" signal, not a rich vocabulary.
- **Ablation:** replacing every message with the average message changes delay by only +0.8% (worse in 7 of 9 episodes, p = 0.33, not significant). The agents can read the messages but barely rely on them.

## RQ1 conclusion (performance and attribution)

E1 differs significantly from both B1 and B2 (p < 0.05), which satisfies RQ1's minimum answer. The direction depends on load: E1 is much better under peak load and slightly worse under light load. However, the ablation shows the advantage **cannot be attributed to the messages themselves**. With one training run per condition, part of the E1 vs B1 gap may come from training-run variance or the extra network capacity of the communication model. This is reported as a finding, which the scope explicitly allows.

## RQ3: generalisation from 2x2 to 4x4

| Scenario | E1 trained on 2x2, run on 4x4 (no retraining) | E1 trained on 4x4 | Tuned | Fixed |
|---|---|---|---|---|
| Uniform | 52.2 | 33.9 | 93.2 | 140.5 |
| Peak | 104.3 | 92.1 | 169.5 | 172.3 |
| Asym | 57.9 | 34.8 | 126.2 | 159.8 |

- With zero retraining, the 2x2 policy still beats tuned fixed-time by 38 to 54% and fixed-time by 39 to 64% on the 4x4 grid (10 of 10 seeds each).
- It is 13 to 67% slower than the policy trained directly on 4x4.
- On the 2x2 grid itself, E1 beats fixed-time by 54% (uniform) and 60% (asym); peak is a tie (73.9 vs 74.6, p = 0.86).
- **Convergence** (`training/fig_training_curves.png`): with the same 500k steps, the 4x4 runs reach 90% of their total improvement within 35k to 73k steps, while 2x2 E1 takes 364k. The 2x2 curve keeps improving slowly to the end. The E1 4x4 run was resumed once at 67k steps; RLlib's reward average restarts there, so the jump at that point is a logging effect, not learning.
- **After about 150k steps, B1 and B2 drift slightly downward while E1 keeps improving.** Final reward per agent is −95k for E1 vs −283k for B1 and −293k for B2.

### Training cost

| Run | Hours | Seconds per iteration |
|---|---|---|
| 4x4 E1 | 3.35 | 12.3 |
| 4x4 B1 | 3.03 | 11.2 |
| 4x4 B2 | 3.49 | 12.9 |
| 2x2 E1 | 1.20 | 4.4 |

Each run is 977 iterations (500k environment steps).

## RQ4: robustness under a 3x demand surge (1200 to 2400 s, never seen in training)

| Controller | Greedy | Sampled |
|---|---|---|
| Fixed-time | **334.6** | **334.6** |
| E1 | 481.9 | 508.8 |
| B1 | 871.2 | 481.6 |
| B2 | 944.7 | 641.5 |
| Tuned | 525.0 | 525.0 |
| Random | 533.3 | 533.3 |

- **Fixed-time is best under extreme overload.** Its long greens give more capacity (Webster's principle); all adaptive controllers switch often and lose time to yellows.
- **Greedy:** E1 is the best adaptive controller. It is 45% lower than B1, 49% lower than B2 and 8% lower than tuned (7 of 10 seeds, significant), but 44% higher than fixed-time.
- **Sampled:** E1, B1, tuned and random are statistically tied. B2 is clearly worst.
- **Degradation:** the policies were never trained on this demand, so this is the graceful-degradation test. In greedy mode E1 has the lowest surge delay of the learned controllers and is the only one that still lets 92% of vehicles in (B1 68%, B2 64%).

## Limitations to state in the report

1. **One training seed per learned condition.** Differences between E1, B1 and B2 include training-run variance.
2. **The messages are not significantly used** (ablation p = 0.33), so the communication benefit is not proven causally.
3. **Only one robustness condition (surge) was run.** The scope lists a second one (single incident).
4. **Fixed and tuned rows come from version 1.** Neither controller is affected by the minimum-green rule; this was verified identical.

## Reproduce

```
python scripts/analyze_4x4.py results/mg5/4x4_eval_mg5_E1.csv results/mg5/4x4_eval_mg5_B1.csv results/mg5/4x4_eval_mg5_B2.csv results/mg5/4x4_eval_mg5_random.csv "results/4x4_eval_hanin.csv:fixed,tuned" --mode greedy --out results/mg5/final/normal_greedy
python scripts/analyze_4x4.py results/mg5/4x4_eval_mg5_explore_E1.csv results/mg5/4x4_eval_mg5_explore_B1.csv results/mg5/4x4_eval_mg5_explore_B2.csv results/mg5/4x4_eval_mg5_random.csv "results/4x4_eval_hanin.csv:fixed,tuned" --mode sampled --out results/mg5/final/normal_sampled
python scripts/compare_modes_4x4.py results/mg5/4x4_eval_mg5_E1.csv results/mg5/4x4_eval_mg5_B1.csv results/mg5/4x4_eval_mg5_B2.csv results/mg5/4x4_eval_mg5_explore_*.csv --out results/mg5/final/greedy_vs_sampled
python scripts/analyze_4x4.py results/mg5/4x4_surge_mg5_E1.csv results/mg5/4x4_surge_mg5_B1.csv results/mg5/4x4_surge_mg5_B2.csv results/mg5/4x4_surge_mg5_fixed.csv results/mg5/4x4_surge_mg5_tuned.csv results/mg5/4x4_surge_mg5_random.csv --mode greedy --scenarios surge --out results/mg5/final/surge_greedy
python scripts/analyze_4x4.py results/mg5/4x4_surge_mg5_explore_E1.csv results/mg5/4x4_surge_mg5_explore_B1.csv results/mg5/4x4_surge_mg5_explore_B2.csv results/mg5/4x4_surge_mg5_fixed.csv results/mg5/4x4_surge_mg5_tuned.csv results/mg5/4x4_surge_mg5_random.csv --mode sampled --scenarios surge --out results/mg5/final/surge_sampled
python scripts/analyze_4x4.py results/mg5/4x4_eval_mg5_E1_from2x2.csv results/mg5/4x4_eval_mg5_E1.csv results/mg5/4x4_eval_mg5_B1.csv "results/4x4_eval_hanin.csv:fixed,tuned" --mode greedy --focus E1_from2x2 --out results/mg5/final/transfer_2x2_to_4x4
python scripts/analyze_4x4.py results/mg5/2x2_eval_mg5_E1.csv results/mg5/2x2_eval_mg5_fixed.csv --mode greedy --out results/mg5/final/grid_2x2
python scripts/plot_training_4x4.py --dir results/mg5 --out results/mg5/final/training
python scripts/interpret_messages_4x4.py --npz results/mg5/e1_messages_4x4_mg5/messages_4x4.npz --report results/mg5/e1_messages_4x4_mg5/message_report.json --out results/mg5/final/messages
```
