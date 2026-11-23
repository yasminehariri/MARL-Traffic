# Final results (version 2: 5 s minimum green, 4x4 grid)

This is the single summary of every result in the project. Each section points to the folder that holds its tables, statistics and figures, and every number below comes from those files.

**How to read the numbers**
- **Metric.** Unless stated otherwise, every number is the mean **true delay per vehicle in seconds** (lower is better), over evaluation (traffic) seeds 0 to 9 in each scenario.
- **Pairing.** Every controller ran on exactly the same route files, so all comparisons are paired by scenario and seed.
- **Statistics.** Paired t-test and exact Wilcoxon signed-rank test, Holm-corrected within each family of comparisons. With 10 seeds the smallest possible Wilcoxon p is 0.002 before correction; after correction it is 0.029 for the 15-comparison main tables.
- **Greedy and sampled.** "Greedy" means the trained agents take their most likely action; "sampled" means they sample from their policy (seeded, so reproducible). Fixed, tuned and random behave the same in both modes.

## The story in brief

1. **In normal traffic, E1 dominates the main comparison at peak hour.** It beats B1, B2, tuned, fixed-time and random in both modes, with 87% less delay than B1 (greedy). It never gridlocked in any of its 180 normal-traffic runs (3 training seeds, 2 modes, 30 episodes each). In light traffic, B1 is a few seconds better.
2. **This is not one lucky training run.** E1 and B1 were each trained three times. In all 15 peak-hour comparisons (5 tests × 3 training seeds), E1 is significantly better, on 10 of 10 traffic seeds each. E1's peak delay stays between 92 and 97 s across its three runs; B1's ranges from 269 to 940 s.
3. **Robustness.**
   - *Sensor failure:* when one junction's detectors break, E1 is the best of E1, B1 and B2 in every scenario and both modes for the main models, and in 8 of 9 cases for the retrained models (the other is a tie).
   - *Demand surge:* under a 3× surge, fixed-time is best overall, and E1 is the best adaptive controller in greedy mode.
4. **Ambulances.** A simple signal-priority rule halves E1's ambulance travel times at no significant cost to the rest of the traffic. At peak, E1's ambulances are much faster than B1's, with or without priority.
5. **The messages carry real information**: the sender's green phase (100% decodable) and its congestion.
6. **But communication is not what makes E1 good.**
   - The no-communication control has exactly E1's network and training, with every incoming message set to zero.
   - It is *better* than E1 in normal traffic in every scenario and both modes (10 of 10 seeds each), and better than every E1 training run. It is also better than B1 in 5 of 6 cases (the sixth is a tie).
   - So E1's large advantage over B1 comes from its much smaller network, not from the messages. In normal traffic the messages slightly hurt.
   - Communication helps in one situation only: a broken junction in greedy mode. There, E1 is 32 to 42% lower than the control, and averaging E1's messages costs it 15 to 16%. In sampled mode the control is better there too.

## Controllers

| Name | What it is | Training |
|---|---|---|
| **E1** | Learned communication: each agent encodes its 9 observations into an 8-number message; each agent receives the mean message of its real neighbours. Network 9→32→8 encoder, 17→64→4 policy (2,014 parameters), plus a diversity penalty that keeps messages distinct | PPO, shared policy, 500k steps, training seeds 0, 1, 2 |
| **B1** | No communication: each agent sees only its own junction (RLlib default network, 256×256) | Same, training seeds 0, 1, 2 |
| **B2** | Hand-crafted communication: neighbours' raw queue lengths appended to the observation | Same, training seed 0 |
| **E1 control** (`E1_nocomm`) | Exactly E1's network, loss and training settings, but the neighbour message it receives is replaced by zeros | Same, training seed 0 |
| Tuned | Fixed cycle, 10 s greens | None |
| Fixed | SUMO's default fixed-time program | None |
| Random | Random phase choice (5 s minimum green kept) | None |

All learned controllers use the same environment: 16 agents, observation of 4 queues, current green phase and time in phase, reward = negative local waiting time, 5 s minimum green, 3 s yellow.

## 1. Normal traffic (RQ1, RQ5)

Folders: `normal_greedy/`, `normal_sampled/`, `greedy_vs_sampled/`; every metric for every controller in `rq5_all_metrics.csv`.

| Scenario | E1 greedy | E1 sampled | B1 greedy | B1 sampled | B2 greedy | B2 sampled | Tuned | Fixed | Random |
|---|---|---|---|---|---|---|---|---|---|
| Uniform | 33.9 | 36.1 | **29.5** | 32.0 | 47.0 | 45.3 | 93.2 | 140.5 | 122.1 |
| Peak | **92.1** | 95.3 | 685.2 | 172.8 | 854.9 | 426.0 | 169.5 | 172.3 | 323.7 |
| Asym | 34.8 | 37.5 | **26.6** | 34.6 | 207.8 | 78.1 | 126.2 | 159.8 | 251.1 |

- **E1 never gridlocks:** 0 teleports in all 60 normal runs (180 including training seeds 1 and 2), and 100% of vehicles entered. B1 gridlocks at peak (10 of 10 episodes greedy, 4 of 10 sampled) and B2 in every scenario.
- **At peak hour, E1 is the best of these six controllers in both modes.** (The no-communication control, section 7, is lower still.)
  - Greedy: 87% lower than B1 and 89% lower than B2.
  - Sampled: 45% lower than B1 and 78% lower than B2.
  - Better than B1 on 10 of 10 seeds in every peak comparison, all significant.
- **In light traffic (uniform, asym), B1 is best.** E1 is 3 to 8 s slower than B1. This is significant except sampled asym (p = 0.19).
- **E1 vs B2:** E1 is lower in every scenario and both modes.
  - Significant at peak (both modes) and in sampled asym.
  - Sampled uniform is significant by Wilcoxon only (t-test p = 0.10).
  - Greedy uniform and greedy asym are not significant, because B2 varies too much between seeds (B2 greedy asym sd = 212 s).
- **B2 (hand-crafted messages) is worse than B1 (no messages)** in every scenario and both modes: 10 of 10 seeds each, all significant.
- **Against fixed-time:**
  - Every learned controller beats it in uniform and asym, except B2 greedy in asym (208 vs 160 s).
  - At peak, only E1 beats it, in both modes (by 45 to 47%).
- **Greedy vs sampled:** E1 changes by only 3 to 8% between modes. B1 and B2 collapse when greedy at peak (B1 from 173 to 685 s, B2 from 426 to 855 s), because the deterministic policy locks into gridlock. E1's policy is the most stable of the three.

## 2. Three training seeds: is it one lucky run?

Folder: `training_seeds/`. E1 and B1 were each trained with seeds 0, 1 and 2 (same settings). Each E1 run is compared with the B1 run of the same training seed, on the same traffic.

| Peak hour, E1 vs B1 | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Normal traffic, greedy | **92.1** vs 685.2 | **96.5** vs 268.7 | **93.2** vs 940.4 |
| Normal traffic, sampled | **95.3** vs 172.8 | **101.1** vs 122.0 | **97.7** vs 213.9 |
| Sensor failure, greedy | **630.9** vs 994.6 | **580.4** vs 835.1 | **588.4** vs 1053.1 |
| Ambulance travel time, priority off | **239.9** vs 787.4 | **234.9** vs 489.4 | **242.0** vs 1077.3 |
| Ambulance travel time, priority on | **98.8** vs 427.0 | **99.6** vs 167.5 | **99.1** vs 723.7 |

(Bold: significantly lower, Wilcoxon, Holm within each test and training seed. All 15 peak cells are significant, E1 lower on 10 of 10 seeds.)

- **E1 is reproducible; B1 is not.** E1's peak delay is 92 to 97 s (greedy) across its three training runs; B1's is 269 to 940 s. The headline advantage at peak holds for every training seed and every test.
- **Light traffic stays B1's.** In normal traffic, uniform and asym, both modes, B1 is significantly lower in 8 of the 12 seed-scenario-mode combinations, and E1 in none; the other 4 are ties. B1 seed 2 is the weakest B1 run: greedy, it gridlocks in 8 of 10 asym episodes (166.6 s).
- **Sensor failure:** E1 is significantly lower than B1 in 8 of the 9 seed-scenario combinations. The exception is seed 1, asym (318.7 vs 323.9 s, no significant difference).
- **Ambulances, light traffic:** without priority, B1's ambulances are slightly faster in uniform and asym for seeds 0 and 1 (by 6 to 19 s). With priority, the two are tied there, apart from seed 1 asym, where B1 is lower by 1.5 s.
- The learning curves of all runs are in `training/` and on the website (`site/data/training.json`).

## 3. Robustness 1: 3× demand surge (RQ4)

Folders: `surge_greedy/`, `surge_sampled/`. Demand triples from 1200 to 2400 s, which the agents never saw in training.

| Controller | Greedy | Sampled |
|---|---|---|
| Fixed-time | **334.6** | **334.6** |
| E1 | 481.9 | 508.8 |
| B1 | 871.2 | 481.6 |
| B2 | 944.7 | 641.5 |
| Tuned | 525.0 | 525.0 |
| Random | 533.3 | 533.3 |

- **Fixed-time is best under extreme overload.** Its long greens give more capacity (Webster's principle), while every adaptive controller switches often and loses time to yellows.
- **Greedy:** E1 is the best adaptive controller.
  - 45% lower than B1, 49% lower than B2 and 8% lower than tuned (7 of 10 seeds, significant).
  - 44% higher than fixed-time.
- **Sampled:** E1, B1, tuned and random are statistically tied; B2 is clearly worst.
- **Graceful degradation:** in greedy mode E1 has the lowest surge delay of the learned controllers, and it is the only one that still lets 92% of vehicles into the network (B1 68%, B2 64%).

## 4. Robustness 2: sensor failure at one junction (RQ4)

Folder: `sensor_failure/`. All 4 queue detectors of light B1 read 0 for the whole episode; every other light works normally. "Fallback" is the standard response to a flagged detector fault: the broken light runs the tuned fixed-time cycle, and every other light keeps its agent.

| Scenario | E1 greedy | E1 sampled | B1 greedy | B1 sampled | B2 greedy | B2 sampled | Control greedy | Control sampled |
|---|---|---|---|---|---|---|---|---|
| Uniform | **348.5** | 95.3 | 577.5 | 257.5 | 476.6 | 209.0 | 597.8 | **52.9** |
| Peak | **630.9** | 284.9 | 994.6 | 552.7 | 966.9 | 512.2 | 935.0 | **196.9** |
| Asym | **241.3** | 94.4 | 366.3 | 134.4 | 388.2 | 190.2 | 355.4 | **42.9** |

(Bold: lowest in each mode.)

- **A single blind junction hurts every learned controller badly.** In both modes, every learned controller gridlocks somewhere in (nearly) every episode.
- **E1 is the best of E1, B1 and B2 in all 12 comparisons** (3 scenarios × 2 modes × 2 opponents): 27 to 63% lower, 10 of 10 seeds each, all significant after Holm correction.
- **The no-communication control behaves very differently in the two modes.**
  - Greedy: it is no better than B1 (597.8 vs 577.5 s uniform, 935.0 vs 994.6 s peak), and E1 is 32 to 42% lower than it (10 of 10 seeds).
  - Sampled: it is the best controller of all, 31 to 55% lower than E1 (E1 is 45 to 120% higher).

  Section 7 discusses what this means.
- **The cost of the failure** (the increase over the same controller in normal traffic), in sampled mode:
  - the control: +14 to +107 s;
  - E1: +57 to +190 s;
  - B1: +100 to +380 s.

  In greedy mode at peak, this comparison is not meaningful, because B1 and B2 are already gridlocked without the failure.
- **Fallback, greedy:**

  | Controller | Uniform | Peak | Asym |
  |---|---|---|---|
  | E1 | 35.3 | **93.6** | 36.6 |
  | B1 | **31.2** | 648.7 | **29.0** |
  | B2 | 52.2 | 840.1 | 223.6 |

  The fallback brings E1 back to its normal level (93.6 vs 92.1 s at peak). With the fallback, the result looks like normal traffic: E1 is far ahead at peak (86% lower than B1), and B1 is a few seconds better in light traffic.
- **The same holds for training seeds 1 and 2** (section 2): E1 is lower than B1 in 8 of 9 seed-scenario combinations.

## 5. Ambulances and signal priority (extension)

Folder: `ambulance/` (training seed 0, four controllers). Seeds 1 and 2 of E1 and B1 are in `training_seeds/`.

**Setup.** 11 ambulances per episode, one every 300 s from 300 to 3300 s, each crossing at least 3 junctions on a boundary-to-boundary shortest path. They are the same for every controller. Every controller ran twice on the same traffic:
- once with ambulances treated like any other car;
- once with **signal priority**: when an ambulance is on a light's approach lane, that light gives its direction green (the nearest ambulance is served first). This is a rule on top of the controller, not something it learned; it never shortens the 3 s yellow or the 5 s minimum green.

**Metric.** Travel time from planned departure to arrival. An ambulance that has not arrived when the episode ends at 3600 s counts with its time so far.

| Controller | Ambulance travel time, off → on (s) | Change | Arrived, off / on (of 330) | Other traffic delay, off → on (s) |
|---|---|---|---|---|
| E1 | 181.2 → **91.7** | -49% | 330 / 330 | 54.6 → 53.3 (-2.4%, n.s.) |
| B1 | 357.8 → 201.3 | -44% | 271 / 294 | 241.6 → 229.4 (-5.1%, n.s.) |
| B2 | 496.6 → 299.0 | -40% | 237 / 263 | 360.3 → 359.8 (-0.1%, n.s.) |
| Tuned | 277.5 → 104.2 | -62% | 316 / 329 | 130.9 → 128.8 (-1.5%, n.s.) |

(All three scenarios together. Per-scenario values and tests are in `ambulance/README.md`.)

- **Priority roughly halves ambulance travel time for every controller.** For E1, stops per trip fall from 5.7 to 0.1, and every ambulance arrives with or without priority.
- **Priority costs the other traffic nothing measurable** in the pooled results (no change is significant). Per scenario, after Holm correction only two changes are significant: E1 at peak, where priority slightly *reduces* the other traffic's delay (-4.7%), and tuned in uniform (+1.2%). B1 in asym rises by 39%, but this is not significant after correction (p = 0.059).
- **At peak, E1's ambulances are much faster than B1's:**
  - without priority: 239.9 vs 787.4 s;
  - with priority: 98.8 vs 427.0 s.

  In B1's peak gridlocks, only 74 of 110 ambulances arrive even with priority. The same holds for training seeds 1 and 2 (section 2).
- **In light traffic without priority, B1's ambulances are slightly faster** (uniform 148.9 vs 155.2 s, asym 137.2 vs 148.4 s, seed 0), matching B1's small light-traffic advantage. With priority the two are tied (90.0 vs 90.4 s and 86.4 vs 86.4 s).

## 6. What the messages contain (RQ2)

Folder: `messages/`.

- The strongest correlation between a message dimension and a traffic variable is |r| = 0.70 (green phase W). Every one of the 8 dimensions has |r| > 0.5 with the sender's current green phase. **This meets the scope's minimum answer (r > 0.5).**
- A receiver can recover the sender's green phase from the message with 100% accuracy (majority baseline 39%), and the sender's total queue with R² = 0.57.
- The t-SNE map splits cleanly by green phase, and the most congested states form their own clusters at the ends of the arms.
- **The messages are effectively one-dimensional:** the first principal component explains 99.9% of the variance (participation ratio 1.0). E1 learned a compact "which phase am I in, how loaded am I" signal, not a rich vocabulary.

## 7. Does communication cause E1's advantage? (RQ1 attribution)

Folders: `messages/` (normal-traffic ablation), `sensor_failure/` (ablation and control under failure), `control/`.

Three tests remove communication in different ways:

**1. Message ablation, normal traffic.** Every message is replaced by the average message, in the already trained E1. Delay changes by only +0.8% (worse in 7 of 9 episodes, not significant). The agents can read the messages but barely rely on them when nothing is broken.

**2. Message ablation, sensor failure.** The same replacement under sensor failure.
- Greedy: E1 with its real messages is 15 to 16% lower than with averaged messages in every scenario (10 of 10 seeds, significant).
- Sampled: 12 to 16% lower, but not significant after correction (7 or 8 of 10 seeds).

So when one junction is blind, the trained E1 does use what its neighbours tell it.

**3. No-communication control.** E1's network, loss and training, but with zeros instead of messages. It is trained from scratch, so it can learn to manage without messages. Its checks are in `tests/test_nocomm.py`: it cannot react to its neighbours, and no gradient reaches the message path.

| | Uniform | Peak | Asym |
|---|---|---|---|
| Normal, greedy: E1 / control / B1 | 33.9 / **27.0** / 29.5 | 92.1 / **86.4** / 685.2 | 34.8 / **26.2** / 26.6 |
| Normal, sampled: E1 / control / B1 | 36.1 / **31.3** / 32.0 | 95.3 / **89.8** / 172.8 | 37.5 / **29.2** / 34.6 |
| Sensor failure, greedy: E1 / control / B1 | **348.5** / 597.8 / 577.5 | **630.9** / 935.0 / 994.6 | **241.3** / 355.4 / 366.3 |
| Sensor failure, sampled: E1 / control / B1 | 95.3 / **52.9** / 257.5 | 284.9 / **196.9** / 552.7 | 94.4 / **42.9** / 134.4 |

(Folder `control/`; Holm correction within each row's 6 tests.)

- **Normal traffic: the control beats E1 everywhere.**
  - In both modes it is lower than E1 in every scenario, on 10 of 10 seeds each (E1 higher by 6 to 33%, all significant).
  - It never gridlocks (0 of 60 runs).
  - It is also lower than every one of E1's three training runs.
  - Against B1, it is significantly lower in 5 of 6 cases (by 87% at peak greedy, 48% at peak sampled), and tied in greedy asym.
- **Sensor failure, greedy: communication helps.** E1 is 32 to 42% lower than the control (10 of 10 seeds in every scenario), while the control is close to B1. Together with the ablation (15 to 16% worse with averaged messages), this shows that in greedy mode E1's messages are what keeps it ahead when a junction is blind.
- **Sensor failure, sampled: the control is best.** It is lower than E1 in every scenario (E1 higher by 45 to 120%; lower on only 0, 2 and 0 of 10 seeds), and far lower than B1 (64 to 80%).

**Conclusion for RQ1.** E1 differs significantly from both B1 and B2 (p < 0.05), which satisfies RQ1's minimum answer.
- Its large advantage over B1 at peak is **not caused by communication.** It comes from what E1 and the control share: a much smaller network (a 17→64→4 policy and a linear value head, where in the control the 8 message inputs are always zero) instead of B1's RLlib default networks with two 256-unit layers, trained with the same PPO settings. In the control the message encoder cannot affect actions, so the diversity loss is not the cause either.
- Communication adds value only in one setting: greedy control with a broken junction. A possible explanation, which we did not test, is that a deterministic policy at the blind junction repeats the same choice, and changing neighbour messages break that pattern, as random sampling does.
- The scope allows this outcome to be reported as a finding: the agents did learn an informative language (section 6), but in this task the benefit of the learned architecture is larger than the benefit of talking.

## 8. Generalisation from 2x2 to 4x4, and training (RQ3)

Folders: `transfer_2x2_to_4x4/`, `grid_2x2/`, `training/`.

| Scenario | E1 trained on 2x2, run on 4x4 (no retraining) | E1 trained on 4x4 | Tuned | Fixed |
|---|---|---|---|---|
| Uniform | 52.2 | 33.9 | 93.2 | 140.5 |
| Peak | 104.3 | 92.1 | 169.5 | 172.3 |
| Asym | 57.9 | 34.8 | 126.2 | 159.8 |

- **Zero-shot transfer:** with no retraining, the 2x2 policy still beats tuned fixed-time by 38 to 54% and fixed-time by 39 to 64% on the 4x4 grid (10 of 10 seeds each). It is 13 to 67% slower than the policy trained directly on 4x4.
- **The 2x2 grid itself:** E1 beats fixed-time by 54% (uniform) and 60% (asym); peak is a tie (73.9 vs 74.6, p = 0.86).
- **Convergence** (`training/fig_training_curves.png`, `training/training_time_table.csv`):
  - The main 4x4 runs reach 90% of their total improvement within 35k to 98k steps, while 2x2 E1 takes 364k.
  - After about 150k steps, B1 and B2 drift slightly downward while E1 keeps improving.
  - Final reward per agent: −95k for E1, −86k for the control, −283k for B1, −293k for B2.
- **Resumed runs:** E1 seed 0 was resumed at 67k steps and E1 seed 2 at 313k. RLlib's reward average restarts at a resume, so a jump there is a logging effect, not learning.

**Training cost** (each run is 977 iterations, 500k environment steps):

| Run | Hours | Seconds per iteration |
|---|---|---|
| 4x4 E1 (seeds 0, 1, 2) | 3.35, 2.99, 3.34 | 12.3, 11.0, 12.3 |
| 4x4 B1 (seeds 0, 1, 2) | 3.03, 2.91, 2.81 | 11.2, 10.7, 10.5 |
| 4x4 B2 | 3.49 | 12.9 |
| 4x4 E1 control | 3.36 | 12.4 |
| 2x2 E1 | 1.20 | 4.4 |

## 9. Reproducibility

Folder: `reproducibility/`. This answers the scope's G6: results must be reproducible within ±5%.

- **Greedy: exactly reproducible.** The repo was cloned fresh from GitHub, and one episode (peak, traffic seed 0) was rerun with each checkpoint stored in the repo:
  - all 7 published models and the control;
  - E1 under sensor failure;
  - E1 with ambulances and priority;
  - the tuned controller.

  Every result is identical to the published one on every metric (`repro_check.csv`). Greedy evaluation is deterministic, also across machines and software versions.
- **Sampled: reproducible on the same software; within a few percent otherwise.** Sampled runs repeat exactly on the same machine (checked twice). The published E1 sampled runs were made on Colab with an older PyTorch. Rerun on the machine and software used for the control, they differ by -1.2% to +6.7% per scenario (`sampled_rerun_check.csv`), because the random draws change between PyTorch versions.
  - All 6 E1 vs control conclusions are unchanged with the rerun values: the control is still lower in every scenario, on 10, 8 and 10 of 10 seeds under failure and 10 of 10 in normal traffic.
  - One cell (sensor failure, asym, sampled) is outside the ±5% band: 94.4 vs 100.7 s.

## 10. Limitations

1. **Light traffic.** E1 is slightly but consistently slower than B1 in uniform and asym traffic (3 to 13 s per vehicle). The control, which is not slower, shows that the messages cause this small loss.
2. **Attribution rests on one control run.** The control was trained once (training seed 0, on CPU rather than the GPU used for E1). Its normal-traffic results are better than all three E1 runs, so the direction is clear, but its own variance across training seeds is unknown. The control differs from B1 essentially in its network size and shape, so that is the likely cause of the gap at peak; a small-network B1 was not trained to confirm it.
3. **Extreme overload.** Under the 3× surge no adaptive controller beats fixed-time.
4. **Training budget.** Each run used 500k steps, half of the 1M planned in the scope, so that all nine runs fit the available compute. Rewards still move by 10 to 40% between the last two 100k-step windows, so longer training could change the size of the gaps, especially for B1, whose runs differ most.
5. **The scope's lane-blocking incident was not run.** Robustness was tested with the demand surge, the sensor failure and the ambulance extension instead.
6. **Fixed and tuned rows come from version 1.** Neither controller is affected by the minimum-green rule; this was verified to be identical.

## 11. Design decisions

- **B2 was trained once (training seed 0); E1 and B1 three times.**
  - The extra seeds were there to test the main claim, E1 against the strongest baseline. That baseline is B1: B2 lost to B1 in every scenario and both modes, on 10 of 10 seeds each.
  - With a fixed compute budget, the extra runs went to E1, B1 and the E1 control, which answer the open questions.
  - B2's ranking therefore rests on one training run.
- **Greedy and sampled are both reported**, because the learned controllers behave very differently in the two modes (section 1).
- **The ambulance priority is a rule, not learned.** This keeps the test about how well each controller's traffic flow supports emergency vehicles, and matches how signal preemption works in practice.

## Folder contents

| Folder | What it answers |
|---|---|
| `normal_greedy/`, `normal_sampled/`, `rq5_all_metrics.csv` | RQ1 and RQ5: all controllers, 3 scenarios, every metric |
| `greedy_vs_sampled/` | How much each learned controller depends on the action mode |
| `training_seeds/` | E1 vs B1 for training seeds 0, 1, 2 on every test |
| `surge_greedy/`, `surge_sampled/` | RQ4: 3× demand surge |
| `sensor_failure/` | RQ4: one blind junction, fallback, message ablation, control |
| `ambulance/` | Extension: ambulances with and without signal priority |
| `messages/` | RQ2: t-SNE, correlation heatmap, temporal plot, normal-traffic ablation |
| `control/` | RQ1 attribution: E1 against the no-communication control |
| `transfer_2x2_to_4x4/`, `grid_2x2/`, `training/` | RQ3: zero-shot transfer, 2x2 results, learning curves and training cost |
| `reproducibility/` | Fresh-clone reruns from the repo checkpoints, and the sampled reruns |

The trained checkpoints of all nine 4x4 runs are in `checkpoints/4x4/` in the repo; the commands to re-run any evaluation are in the main README.

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
python scripts/analyze_ambulance.py --out results/mg5/final/ambulance
python scripts/analyze_extensions.py          # sensor_failure/, training_seeds/, control/
python scripts/export_site_data.py --check    # the website's data files
```
