import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.traci_env import TraCIEnv
import traci

env = TraCIEnv()
obs = env.reset(route_file='configs/routes.rou.xml')
print('Scenario loader OK — agents:', list(obs.keys()))
traci.close()
print('PASSED')