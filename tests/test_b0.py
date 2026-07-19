import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.traci_env import TraCIEnv
from env.fixed_time_controller import FixedTimeController
from env.metrics_logger import MetricsLogger
from env.traci_env import LANE_IDS
import traci

NUM_EPISODES = 10

SCENARIOS = [
    ("uniform",  "configs/routes.rou.xml"),
    ("peak",     "configs/routes_peak.rou.xml"),
    ("asymmetric", "configs/routes_asym.rou.xml"),
]

env = TraCIEnv()
controller = FixedTimeController()

for scenario_name, route_file in SCENARIOS:
    print(f"\n--- Running B0 on {scenario_name} ---")
    logger = MetricsLogger(
        log_dir="logs",
        run_id=f"b0_{scenario_name}_final",
        condition="B0",
        scenario=scenario_name
    )

    for episode in range(1, NUM_EPISODES + 1):
        obs = env.reset(route_file=route_file)
        controller.reset()
        done = False

        while not done:
            actions = controller.get_actions()
            obs, reward, done, info = env.step(actions)
            logger.log_step(obs, reward, LANE_IDS)

        traci.close()
        logger.log_episode(episode)

print("\nB0 complete across all 3 scenarios — check logs/results.csv")