from env.traci_env import TraCIEnv
import traci

env = TraCIEnv()

# Launch SUMO and reset
obs = env.reset()
assert len(obs) == 4, f'Expected 4 agents, got {len(obs)}'
for tl_id, o in obs.items():
    assert len(o) == 9, f'Expected 9-dim obs for {tl_id}, got {len(o)}'
print('reset() OK —', list(obs.keys()))

# Run one real step
actions = [0, 0, 0, 0]
obs, reward, done, info = env.step(actions)
assert len(obs) == 4
assert isinstance(reward, float)
assert isinstance(done, bool)
print('step() OK — reward:', reward, ' done:', done)

traci.close()
print('All checks passed: shapes correct, SUMO pipeline live.')
