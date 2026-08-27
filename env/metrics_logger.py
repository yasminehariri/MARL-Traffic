import csv
import os
import xml.etree.ElementTree as ET


RESULTS_COLUMNS = [
    "run_id", "condition", "scenario", "episode",
    "avg_wait_time", "avg_queue_length", "throughput", "cumulative_reward"
]


class MetricsLogger:
    def __init__(self, log_dir="logs", run_id="run", condition="B0", scenario="uniform"):
        self.log_dir = log_dir
        self.run_id = run_id
        self.condition = condition
        self.scenario = scenario
        os.makedirs(log_dir, exist_ok=True)

        self.results_path = os.path.join(log_dir, "results.csv")
        self._init_csv()

        # Per-episode accumulators
        self._wait_times = []
        self._queue_lengths = []
        self._cumulative_reward = 0.0
        self._steps = 0

    def _init_csv(self):
        # Only write header if file doesn't exist yet
        if not os.path.exists(self.results_path):
            with open(self.results_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
                writer.writeheader()

    def log_step(self, obs, reward, lane_ids):
    for tl_id, vec in obs.items():
        queues = vec[0:4]  # all 4 real queue values
        self._queue_lengths.append(sum(queues) / len(queues))

    self._cumulative_reward += reward
    self._steps += 1

    def log_episode(self, episode, tripinfo_path="configs/tripinfo.xml"):
        # Called at end of each episode
        avg_wait, throughput = self._parse_tripinfo(tripinfo_path)
        avg_queue = sum(self._queue_lengths) / len(self._queue_lengths) if self._queue_lengths else 0.0

        row = {
            "run_id": self.run_id,
            "condition": self.condition,
            "scenario": self.scenario,
            "episode": episode,
            "avg_wait_time": round(avg_wait, 4),
            "avg_queue_length": round(avg_queue, 4),
            "throughput": throughput,
            "cumulative_reward": round(self._cumulative_reward, 4),
        }

        with open(self.results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
            writer.writerow(row)

        print(f"Episode {episode} — wait: {avg_wait:.2f}s | queue: {avg_queue:.4f} | "
              f"throughput: {throughput} | reward: {self._cumulative_reward:.2f}")

        # Reset accumulators
        self._wait_times = []
        self._queue_lengths = []
        self._cumulative_reward = 0.0
        self._steps = 0

    def _parse_tripinfo(self, path):
        if not os.path.exists(path):
            return 0.0, 0

        tree = ET.parse(path)
        root = tree.getroot()

        wait_times = []
        completed = 0

        for trip in root.findall("tripinfo"):
            vaporized = trip.attrib.get("vaporized", "")
            if vaporized == "":
                completed += 1
                wait_times.append(float(trip.attrib.get("waitingTime", 0)))

        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
        return avg_wait, completed