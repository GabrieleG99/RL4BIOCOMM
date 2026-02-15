from typing import Dict

import numpy as np

class MoveEnvWrapper:

    EXPOSED_ATTRIBUTES = [
        "encoder",
        "decoder",
        "agents",
        "n_iters",
        "n_agents",
        "mol_types",
        "agents_dis",
        "sender",
        "receiver"
    ]

    def __init__(self, env):
        self.env = env

    def plot_graph(self):
        self.env.plot_graph()

    def set_deterministic(self, mode: bool=True):
        self.env.set_deterministic(mode)

    def step(self, action: Dict[str, np.ndarray]):

        agents_moves = action["move"]

        for i, move in enumerate(agents_moves):
            self.env.agents[i].move(move)

        obs, reward, done, info = self.env.step(action["message"])
        return obs, reward, done, info

    def __getattr__(self, item):

        if item in self.EXPOSED_ATTRIBUTES:
            return getattr(self.env, item)
        elif item in self.__dict__:
            return getattr(self, item)
        else:
            raise AttributeError(f"No attribute found: {item}")

    def reset(self, X_batch, y_batch):
        obs = self.env.reset(X_batch, y_batch)
        return obs