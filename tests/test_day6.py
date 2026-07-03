import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.traci_env import TraCIEnv

env = TraCIEnv()
obs = env.reset()
for tl_id, vec in obs.items():
    print(tl_id, len(vec), vec)

obs, reward, done, info = env.step([0, 0, 0, 0])
print("reward:", reward)

import traci
traci.close()