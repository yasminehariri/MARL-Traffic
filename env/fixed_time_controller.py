from env.traci_env import TL_IDS, GREEN_PHASES
import traci

# B0 baseline: fixed-time control
# 30s green per phase, cycles through all 4 green phases
# No learning, no observation, no reward signal

PHASE_DURATION = 30  # seconds per green phase


class FixedTimeController:
    def __init__(self):
        self.agents = TL_IDS
        self.phase_index = {tl: 0 for tl in self.agents}
        self.elapsed = {tl: 0 for tl in self.agents}

    def reset(self):
        self.phase_index = {tl: 0 for tl in self.agents}
        self.elapsed = {tl: 0 for tl in self.agents}

    def get_actions(self):
        actions = []
        for tl in self.agents:
            self.elapsed[tl] += 1
            if self.elapsed[tl] >= PHASE_DURATION:
                self.elapsed[tl] = 0
                self.phase_index[tl] = (self.phase_index[tl] + 1) % len(GREEN_PHASES)
            actions.append(self.phase_index[tl])
        return actions