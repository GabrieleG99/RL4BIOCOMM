from typing import Dict

import numpy as np

class MoveEnvWrapper:
    """
    Wrapper that adds movement support to a ``VectorizedEnvironment``.

    Before forwarding message actions to the underlying environment's
    ``step`` method, this wrapper applies per-agent position deltas
    to the vectorised positions array.

    Expected action format::

        action = {
            "move":    np.ndarray (n_agents, n_envs, 2),
            "message": np.ndarray (n_agents, n_envs, msg_dim),
        }
    """

    EXPOSED_ATTRIBUTES = [
        "encoder",
        "decoder",
        "agents",
        "n_envs",
        "n_iters",
        "n_agents",
        "mol_types",
        "agents_dis",
        "sender",
        "receiver",
        "positions",
    ]

    def __init__(self, env):
        self.env = env

    def plot_graph(self, env_idx: int = 0):
        self.env.plot_graph(env_idx=env_idx)

    def set_deterministic(self, mode: bool = True):
        self.env.set_deterministic(mode)

    def step(self, action: Dict[str, np.ndarray]):
        # action["move"]: (n_agents, n_envs, 2)
        self.env.update_positions(action["move"])

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