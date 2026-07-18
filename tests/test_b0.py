import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.traci_env import TraCIEnv
from env.fixed_time_controller import FixedTimeController
from env.metrics_logger import MetricsLogger
from env.traci_env import LANE_IDS
import traci

env = TraCIEnv()
controller = FixedTimeController()
logger = MetricsLogger(
    log_dir="logs",
    run_id="b0_uniform_final",
    condition="B0",
    scenario="uniform"
)

NUM_EPISODES = 10

for episode in range(1, NUM_EPISODES + 1):
    obs = env.reset()
    controller.reset()
    done = False
    step = 0

    while not done:
        actions = controller.get_actions()
        obs, reward, done, info = env.step(actions)
        logger.log_step(obs, reward, LANE_IDS)
        step += 1

    traci.close()
    logger.log_episode(episode)
    print(f"Episode {episode} complete — {step} steps")

print("B0 test done — check logs/results.csv")