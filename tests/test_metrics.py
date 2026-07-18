import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.traci_env import TraCIEnv
from env.metrics_logger import MetricsLogger
from env.traci_env import LANE_IDS
import traci

env = TraCIEnv()
logger = MetricsLogger(
    log_dir="logs",
    run_id="test_001",
    condition="B0",
    scenario="uniform"
)

for episode in range(1, 3):
    obs = env.reset()
    done = False
    step = 0

    while not done and step < 100:
        actions = [0, 0, 0, 0]
        obs, reward, done, info = env.step(actions)
        logger.log_step(obs, reward, LANE_IDS)
        step += 1

    # Close SUMO first so tripinfo.xml is fully written
    traci.close()

    # Now parse tripinfo
    logger.log_episode(episode)

print("MetricsLogger test complete — check logs/results.csv")