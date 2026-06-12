import multiprocessing as mp
import traceback
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

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


class MultiProcessEnvWrapper:

    def __init__(self,
                 env_builder: Callable[..., Any],
                 n_envs: int,
                 n_workers: int,
                 seed: int=42,
                 start_method: str="fork"):

        self.env_builder = env_builder
        self.n_envs = n_envs
        self.n_workers = min(n_workers, n_envs)
        self.closed = False

        if n_envs < 1:
            raise ValueError("n_envs must be at least 1")
        if n_workers < 1:
            raise ValueError("n_workers must be at least 1")

        ctx = mp.get_context(start_method)
        env_indices = np.array_split(np.arange(n_envs), self.n_workers)
        self._workers = []
        self._pipes = []

        for worker_indices in env_indices:
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=_env_worker,
                args=(child_pipe, env_builder, worker_indices.tolist(), seed),
            )
            process.daemon = True
            process.start()
            child_pipe.close()
            self._pipes.append(parent_pipe)
            self._workers.append(process)

        try:
            self.n_agents = self.get_attr("n_agents")
        except Exception:
            self.close()
            raise

    def reset(self, *args, **kwargs):
        payloads = self._split_call(args, kwargs)
        results = self._dispatch("reset", payloads)
        return self._stack(results)

    def step(self, actions):
        payloads = self._split_values(actions)
        results = self._dispatch("step", payloads)
        return self._stack(results)

    def set_deterministic(self, mode: bool=True):
        self.call("set_deterministic", mode)

    def reset_rng_state(self):
        self.call("reset_rng_state")

    def call(self, method_name: str, *args, **kwargs):
        payloads = [(args, kwargs) for _ in range(self.n_envs)]
        return self._dispatch("call", (method_name, payloads))

    def get_attr(self, name: str):
        self._ensure_open()
        self._pipes[0].send(("get_attr", name))
        return self._recv_one(self._pipes[0])

    def close(self):
        if self.closed:
            return

        for pipe in self._pipes:
            if not pipe.closed:
                try:
                    pipe.send(("close", None))
                except (BrokenPipeError, EOFError):
                    pass

        for process in self._workers:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

        for pipe in self._pipes:
            pipe.close()

        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(f"No attribute found: {item}")
        try:
            return self.get_attr(item)
        except RuntimeError as exc:
            raise AttributeError(f"No attribute found: {item}") from exc

    def _dispatch(self, command: str, payload: Any):
        self._ensure_open()
        worker_payloads = self._chunk_payload(payload)

        for pipe, worker_payload in zip(self._pipes, worker_payloads):
            pipe.send((command, worker_payload))

        results = []
        for pipe in self._pipes:
            results.extend(self._recv_one(pipe))
        return results

    def _chunk_payload(self, payload: Any) -> List[Any]:
        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], str):
            method_name, per_env_payloads = payload
            return [
                (method_name, list(chunk))
                for chunk in self._chunks(per_env_payloads)
            ]

        return [list(chunk) for chunk in self._chunks(payload)]

    def _chunks(self, values: Sequence[Any]) -> Iterable[Sequence[Any]]:
        start = 0
        for pipe_idx in range(self.n_workers):
            size = self.n_envs // self.n_workers
            if pipe_idx < self.n_envs % self.n_workers:
                size += 1
            yield values[start:start + size]
            start += size

    def _recv_one(self, pipe):
        try:
            status, payload = pipe.recv()
        except EOFError as exc:
            raise RuntimeError("Environment worker exited unexpectedly") from exc
        if status == "ok":
            return payload
        raise RuntimeError(payload)

    def _split_call(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        return [
            (
                tuple(self._split_value(arg, env_idx) for arg in args),
                {key: self._split_value(value, env_idx) for key, value in kwargs.items()},
            )
            for env_idx in range(self.n_envs)
        ]

    def _split_values(self, value: Any):
        return [self._split_value(value, env_idx) for env_idx in range(self.n_envs)]

    def _split_value(self, value: Any, env_idx: int):
        if isinstance(value, dict):
            return {key: self._split_value(item, env_idx) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._split_value(item, env_idx) for item in value)
        if isinstance(value, list):
            return [self._split_value(item, env_idx) for item in value]
        if isinstance(value, np.ndarray):
            if value.ndim >= 2 and value.shape[1] == self.n_envs:
                return value[:, env_idx, ...]
            if value.ndim >= 1 and value.shape[0] == self.n_envs:
                return value[env_idx, ...]
        return value

    def _stack(self, values: Sequence[Any]):
        first = values[0]

        if isinstance(first, tuple):
            return tuple(self._stack([value[idx] for value in values]) for idx in range(len(first)))

        if isinstance(first, list):
            return [self._stack([value[idx] for value in values]) for idx in range(len(first))]

        if isinstance(first, dict):
            return {
                key: self._stack([value[key] for value in values])
                for key in first
            }

        if isinstance(first, np.ndarray):
            arrays = [np.asarray(value) for value in values]
            if first.ndim >= 1 and first.shape[0] == self.n_agents:
                return np.stack(arrays, axis=1)
            return np.stack(arrays, axis=0)

        if np.isscalar(first):
            return np.asarray(values)

        return list(values)

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("MultiProcessEnvWrapper is closed")


def _env_worker(pipe, env_builder: Callable[..., Any], env_indices: List[int], seed: int):
    try:
        envs = [env_builder(seed=seed + env_idx) for env_idx in env_indices]

        while True:
            command, payload = pipe.recv()

            if command == "close":
                for env in envs:
                    if hasattr(env, "close"):
                        env.close()
                pipe.send(("ok", None))
                break

            if command == "reset":
                result = [
                    env.reset(*args, **kwargs)
                    for env, (args, kwargs) in zip(envs, payload)
                ]
                pipe.send(("ok", result))
                continue

            if command == "step":
                result = [env.step(action) for env, action in zip(envs, payload)]
                pipe.send(("ok", result))
                continue

            if command == "call":
                method_name, call_payloads = payload
                result = [
                    getattr(env, method_name)(*args, **kwargs)
                    for env, (args, kwargs) in zip(envs, call_payloads)
                ]
                pipe.send(("ok", result))
                continue

            if command == "get_attr":
                pipe.send(("ok", getattr(envs[0], payload)))
                continue

            raise ValueError(f"Unknown worker command: {command}")

    except EOFError:
        pass
    except Exception:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()
