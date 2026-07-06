import sys, os
sys.path.insert(0, os.path.abspath('.'))
from env.traci_env import TraCIEnv
import traci

env = TraCIEnv()
obs = env.reset()
obs, reward, done, info = env.step([0, 0, 0, 0])

assert len(obs) == 4, 'FAIL: wrong number of agents'
for tl, vec in obs.items():
    assert len(vec) == 9, 'FAIL: wrong obs dim for ' + tl
assert isinstance(reward, float), 'FAIL: reward not float'
assert isinstance(done, bool), 'FAIL: done not bool'

traci.close()

logs_exist = all(os.path.exists('logs/run_' + str(i) + '.csv') for i in range(1, 6))

print('EXIT CRITERION: PASSED')
print('Agents:', list(obs.keys()))
print('Obs dim:', len(list(obs.values())[0]))
print('Reward:', reward)
print('5 runs logged:', logs_exist)