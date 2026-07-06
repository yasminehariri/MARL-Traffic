import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import traci
from env.traci_env import TraCIEnv

NUM_RUNS = 5
LOG_DIR = "logs"
MAX_STEPS = 3600

env = TraCIEnv()

for run in range(NUM_RUNS):
    obs = env.reset()
    log_path = os.path.join(LOG_DIR, f"run_{run+1}.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "agent",
            "q0", "q1", "q2", "q3",
            "phase_0", "phase_1", "phase_2", "phase_3",
            "elapsed_ratio", "reward"
        ])

        for step in range(MAX_STEPS):
            actions = [0, 0, 0, 0]
            obs, reward, done, info = env.step(actions)

            for agent, vec in obs.items():
                writer.writerow([
                    step, agent,
                    vec[0], vec[1], vec[2], vec[3],
                    vec[4], vec[5], vec[6], vec[7],
                    vec[8], reward
                ])

            if done:
                break

    print(f"Run {run+1} logged to {log_path}")

traci.close()
print("All 5 runs complete.")