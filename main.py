import argparse
import json
import math
import random
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from gymnasium.spaces import Box, MultiDiscrete
from torch import nn

from algorithms.MAPPO import MAPPO
from envs.environments import Environment
from envs.env_wrappers import MultiProcessEnvWrapper
from learners.learners import MAPPOLearner
from models.Agent import Agent
from models.layers import FeedForwardNN, RecurrentNN
from models.policies import ActorCriticPolicy
from utils.utils import (
    get_classification_data,
    get_classification_data_from_ds,
    get_dataloader,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_dataset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    source = cfg.get("source", "uci")
    if source == "uci":
        X_train, X_val, X_test, y_train, y_val, y_test = get_classification_data_from_ds(
            dataset_name=cfg.get("name", "iris"),
            test_size=cfg.get("test_size", 0.2),
            val_size=cfg.get("val_size", 0.2),
            random_state=cfg.get("random_state", 42),
        )
    elif source == "synthetic":
        X_train, X_val, X_test, y_train, y_val, y_test = get_classification_data(
            n_samples=cfg.get("n_samples", 500),
            n_features=cfg.get("n_features", 8),
            n_classes=cfg.get("n_classes", 3),
            val_size=cfg.get("val_size", 0.2),
            test_size=cfg.get("test_size", 0.2),
            random_state=cfg.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unsupported dataset source: {source}")

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    n_classes = int(len(np.unique(y_train)))

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "n_classes": n_classes,
        "input_dim": int(X_train.shape[1]),
    }


def build_loaders(data: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
    val_loader = get_dataloader(data["X_val"], data["y_val"], batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(data["X_test"], data["y_test"], batch_size=batch_size, shuffle=False)
    return {"val_loader": val_loader, "test_loader": test_loader}


def load_or_init_models(
    dataset_name: str,
    input_dim: int,
    obs_dim: int,
    n_classes: int,
    encoder_hidden: List[int],
    decoder_hidden: List[int],
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    pretrain_cfg: Dict[str, Any],
    device: str,
) -> Tuple[FeedForwardNN, FeedForwardNN]:
    """
    Try to load pretrained encoder/classification-head checkpoints.
    If checkpoints are missing or incompatible, train them with the notebook flow:
    autoencoder pretraining followed by frozen-encoder classifier training.
    """
    model_save_path = Path(pretrain_cfg.get("model_save_path", "best_models"))
    model_save_path.mkdir(parents=True, exist_ok=True)
    encoder_path = model_save_path / f"best_encoder_{dataset_name}.pt"
    ae_decoder_path = model_save_path / f"best_decoder_{dataset_name}.pt"
    decoder_path = model_save_path / f"best_ch_{dataset_name}.pt"
    encoder_activation = pretrain_cfg.get("encoder_activation", "relu")
    decoder_activation = pretrain_cfg.get("decoder_activation", "relu")
    use_batchnorm = pretrain_cfg.get("use_batchnorm", True)

    def fresh_models() -> Tuple[FeedForwardNN, FeedForwardNN]:
        encoder = FeedForwardNN(
            input_size=input_dim,
            hidden_size=encoder_hidden,
            output_size=obs_dim,
            activation=encoder_activation,
            use_batchnorm=use_batchnorm,
        ).to(device)
        decoder = FeedForwardNN(
            input_size=obs_dim,
            hidden_size=decoder_hidden,
            output_size=n_classes,
            activation=decoder_activation,
            use_batchnorm=use_batchnorm,
        ).to(device)
        return encoder, decoder

    def freeze_models(encoder: FeedForwardNN, decoder: FeedForwardNN) -> Tuple[FeedForwardNN, FeedForwardNN]:
        encoder.eval()
        decoder.eval()
        for model in (encoder, decoder):
            for param in model.parameters():
                param.requires_grad = False
        return encoder, decoder

    if encoder_path.exists() and decoder_path.exists():
        encoder = FeedForwardNN.load(str(encoder_path), device)
        decoder = FeedForwardNN.load(str(decoder_path), device)
        if (
            encoder.input_size == input_dim
            and encoder.output_size == obs_dim
            and decoder.input_size == obs_dim
            and decoder.output_size == n_classes
        ):
            return freeze_models(encoder, decoder)
        print("WARNING: pretrained encoder/decoder dimensions do not match config; retraining them")

    encoder, ae_decoder = _train_autoencoder(
        dataset_name=dataset_name,
        input_dim=input_dim,
        obs_dim=obs_dim,
        hidden_size=encoder_hidden,
        activation=encoder_activation,
        use_batchnorm=use_batchnorm,
        X_train=X_train,
        X_val=X_val,
        batch_size=batch_size,
        lr=pretrain_cfg.get("ae_lr", pretrain_cfg.get("lr", 3e-4)),
        epochs=pretrain_cfg.get("ae_epochs", pretrain_cfg.get("epochs", 10000)),
        patience=pretrain_cfg.get("ae_patience", pretrain_cfg.get("patience", 0)),
        min_delta=pretrain_cfg.get("min_delta", 0.0),
        encoder_path=encoder_path,
        decoder_path=ae_decoder_path,
        device=device,
    )
    del ae_decoder

    decoder = _train_classification_head(
        dataset_name=dataset_name,
        encoder=encoder,
        obs_dim=obs_dim,
        n_classes=n_classes,
        hidden_size=decoder_hidden,
        activation=decoder_activation,
        use_batchnorm=use_batchnorm,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        batch_size=batch_size,
        lr=pretrain_cfg.get("classifier_lr", pretrain_cfg.get("lr", 3e-4)),
        epochs=pretrain_cfg.get("classifier_epochs", pretrain_cfg.get("epochs", 10000)),
        patience=pretrain_cfg.get("classifier_patience", pretrain_cfg.get("patience", 0)),
        min_delta=pretrain_cfg.get("min_delta", 0.0),
        decoder_path=decoder_path,
        device=device,
    )

    return freeze_models(FeedForwardNN.load(str(encoder_path), device), decoder)


def _train_autoencoder(
    dataset_name: str,
    input_dim: int,
    obs_dim: int,
    hidden_size: List[int],
    activation: str,
    use_batchnorm: bool,
    X_train: np.ndarray,
    X_val: np.ndarray,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    min_delta: float,
    encoder_path: Path,
    decoder_path: Path,
    device: str,
) -> Tuple[FeedForwardNN, FeedForwardNN]:
    encoder = FeedForwardNN(
        input_size=input_dim,
        output_size=obs_dim,
        hidden_size=hidden_size,
        activation=activation,
        use_batchnorm=use_batchnorm,
    ).to(device)
    decoder = FeedForwardNN(
        input_size=obs_dim,
        output_size=input_dim,
        hidden_size=hidden_size,
        activation=activation,
        use_batchnorm=use_batchnorm,
    ).to(device)
    train_loader = get_dataloader(X_train.astype(np.float32), X_train.astype(np.float32), batch_size=batch_size)
    val_loader = get_dataloader(X_val.astype(np.float32), X_val.astype(np.float32), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")
    stale_epochs = 0

    print(f"Training autoencoder checkpoints for {dataset_name}")
    for epoch in range(1, int(epochs) + 1):
        encoder.train()
        decoder.train()
        train_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = decoder(encoder(x))
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        encoder.eval()
        decoder.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                val_losses.append(loss_fn(decoder(encoder(x)), y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            stale_epochs = 0
            encoder.save(str(encoder_path))
            decoder.save(str(decoder_path))
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 100 == 0:
            print(f"AE epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if patience > 0 and stale_epochs >= patience:
            print(f"AE early stopping at epoch {epoch}; best_val_loss={best_val_loss:.4f}")
            break

    return FeedForwardNN.load(str(encoder_path), device), FeedForwardNN.load(str(decoder_path), device)


def _train_classification_head(
    dataset_name: str,
    encoder: FeedForwardNN,
    obs_dim: int,
    n_classes: int,
    hidden_size: List[int],
    activation: str,
    use_batchnorm: bool,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    min_delta: float,
    decoder_path: Path,
    device: str,
) -> FeedForwardNN:
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    classification_head = FeedForwardNN(
        input_size=obs_dim,
        output_size=n_classes,
        hidden_size=hidden_size,
        activation=activation,
        use_batchnorm=use_batchnorm,
    ).to(device)
    train_loader = get_dataloader(X_train, y_train, batch_size=batch_size)
    val_loader = get_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(classification_head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_val_acc = -float("inf")
    stale_epochs = 0

    print(f"Training classification head checkpoint for {dataset_name}")
    for epoch in range(1, int(epochs) + 1):
        classification_head.train()
        train_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.long().to(device)
            with torch.no_grad():
                encoded = encoder(x)
            logits = classification_head(encoded)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        classification_head.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.long().to(device)
                logits = classification_head(encoder(x))
                val_losses.append(loss_fn(logits, y).item())
                predicted = logits.argmax(dim=1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_acc = correct / total if total > 0 else 0.0

        if val_acc >= best_val_acc or val_loss < best_val_loss - min_delta:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            stale_epochs = 0
            classification_head.save(str(decoder_path))
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Classifier epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )

        if patience > 0 and stale_epochs >= patience:
            print(
                f"Classifier early stopping at epoch {epoch}; "
                f"best_val_loss={best_val_loss:.4f}, best_val_acc={best_val_acc:.3f}"
            )
            break

    return FeedForwardNN.load(str(decoder_path), device)


def build_spaces(action_dim: int, obs_dim: int, n_agents: int, max_mol_out: int) -> tuple[Dict[str, MultiDiscrete | Box], Dict[str, MultiDiscrete | Box]]:
    """
    Actions follow notebook style: MultiDiscrete with one bucket per molecule type.
    Receiver/relay observations are MultiDiscrete over aggregated incoming molecules.
    """

    if action_dim != obs_dim:
        print("WARNING: action_dim != obs_dim, converting to the same size")
        obs_dim = action_dim

    action_space = {
        "sender": MultiDiscrete([max_mol_out] * action_dim),
        "receiver": MultiDiscrete([max_mol_out] * action_dim),
        "relay": MultiDiscrete([max_mol_out] * action_dim),
    }

    # Each entry in obs MultiDiscrete counts molecules from all agents; sender sees encoded input.
    obs_space = {
        "sender": Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        "receiver": MultiDiscrete([max_mol_out * n_agents] * action_dim),
        "relay": MultiDiscrete([max_mol_out * n_agents] * action_dim),
    }
    return action_space, obs_space


def build_agents(n_agents: int, action_dim: int, device: str) -> List[Agent]:
    return [Agent(max_mol_out=action_dim, n_mol_types=action_dim, device=device) for _ in range(n_agents)]


def build_environment(
    env_cfg: Dict[str, Any],
    action_space: Dict[str, Any],
    obs_space: Dict[str, Any],
    encoder: FeedForwardNN,
    decoder: FeedForwardNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    action_dim: int,
    device: str,
    space_shape: Tuple[int, int],
    layout_seed: int,
    seed: int,
) -> Environment:
    env = Environment(
        agents=build_agents(env_cfg["n_agents"], action_dim, device),
        policy_action_space=action_space,
        policy_obs_space=obs_space,
        space_shape=space_shape,
        n_iters=env_cfg.get("n_iters", 3),
        mol_types=env_cfg.get("mol_types", action_dim),
        noisy=env_cfg.get("noisy", True),
        shared_obs=env_cfg.get("shared_obs", True),
        sr_choice=env_cfg.get("sr_choice", "furthest"),
        is_continuous=env_cfg.get("is_continuous", True),
        encoder=encoder,
        decoder=decoder,
        X_train=X_train,
        y_train=y_train,
        device=device,
        max_step_count=env_cfg.get("max_step_count", 30),
        seed=layout_seed,
    )
    env.seed_value = seed
    env.rng = np.random.RandomState(seed)
    return env


def partition_relay_agents(env: Environment, n_partitions: int, method: str) -> Dict[int, int]:
    sender_idx = env.sender
    receiver_idx = env.receiver
    relay_indices = [i for i in range(env.n_agents) if i not in (sender_idx, receiver_idx)]
    n_relays = len(relay_indices)

    if n_partitions > n_relays:
        raise ValueError(f"n_partitions ({n_partitions}) must be <= number of relays ({n_relays})")

    positions = np.array([agent.state.position for agent in env.agents])

    if method == "distance_based":
        sender_pos = positions[sender_idx]
        receiver_pos = positions[receiver_idx]
        distances = []
        for rid in relay_indices:
            relay_pos = positions[rid]
            dist_from_sender = np.linalg.norm(relay_pos - sender_pos)
            dist_from_receiver = np.linalg.norm(relay_pos - receiver_pos)
            total = dist_from_sender + dist_from_receiver + 1e-8
            distances.append((rid, dist_from_sender / total))
        distances.sort(key=lambda x: x[1])
        partition_size = max(1, n_relays // n_partitions)
        mapping = {rid: min(i // partition_size, n_partitions - 1) for i, (rid, _) in enumerate(distances)}
    elif method == "random":
        shuffled = relay_indices.copy()
        np.random.shuffle(shuffled)
        partition_size = max(1, n_relays // n_partitions)
        mapping = {rid: min(i // partition_size, n_partitions - 1) for i, rid in enumerate(shuffled)}
    else:
        raise ValueError(f"Unknown partitioning method: {method}")
    return mapping


def build_agent_policy_mapping(env: Environment, partition_mapping: Dict[int, int]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for i in range(env.n_agents):
        if i == env.sender:
            mapping[i] = "sender"
        elif i == env.receiver:
            mapping[i] = "receiver"
        else:
            mapping[i] = f"relay_partition_{partition_mapping[i]}"
    return mapping


def build_policies(
    policy_cfg: Dict[str, Any],
    action_space: Dict[str, Any],
    obs_space: Dict[str, Any],
    partition_mapping: Dict[int, int],
    critic_input_size: int,
    device: str,
) -> Dict[str, ActorCriticPolicy]:
    FEAT_EX_CLAZZ = {
        "mlp": FeedForwardNN,
        "recurrent": RecurrentNN,
    }

    net_arch = {
        "pi": policy_cfg.get("pi_arch", [16, 16]),
        "vf": policy_cfg.get("vf_arch", [16, 16]),
    }
    feat_kwargs = {
        "hidden_size": policy_cfg.get("hidden_sizes", []),
        "output_size": policy_cfg.get("feature_dim", 16),
    }
    optim_kwargs = {
        "actor": {"lr": policy_cfg.get("actor_lr", 3e-4), "weight_decay": policy_cfg.get("weight_decay", 0.0)},
        "critic": {"lr": policy_cfg.get("critic_lr", policy_cfg.get("actor_lr", 3e-4)), "weight_decay": policy_cfg.get("weight_decay", 0.0)},
    }

    n_partitions = len(set(partition_mapping.values()))
    policies: Dict[str, ActorCriticPolicy] = {}

    feat_clazz = policy_cfg.get("features_extractor_class", "mlp")
    feat_clazz = FEAT_EX_CLAZZ.get(feat_clazz, None)

    policies["sender"] = ActorCriticPolicy(
        action_space=action_space["sender"],
        observation_space=obs_space["sender"],
        optim_kwargs=optim_kwargs,
        features_extractor_class=feat_clazz,
        features_extractor_kwargs=feat_kwargs,
        shared_critic_input_size=critic_input_size,
        net_arch=net_arch,
        activation=policy_cfg.get("activation", "relu"),
        use_popart=policy_cfg.get("use_popart", False),
        device=device,
    )

    policies["receiver"] = ActorCriticPolicy(
        action_space=action_space["receiver"],
        observation_space=obs_space["receiver"],
        optim_kwargs=optim_kwargs,
        features_extractor_class=feat_clazz,
        features_extractor_kwargs=feat_kwargs,
        shared_critic_input_size=critic_input_size,
        net_arch=net_arch,
        activation=policy_cfg.get("activation", "relu"),
        use_popart=policy_cfg.get("use_popart", False),
        device=device,
    )

    for partition_id in range(n_partitions):
        policies[f"relay_partition_{partition_id}"] = ActorCriticPolicy(
            action_space=action_space["relay"],
            observation_space=obs_space["relay"],
            optim_kwargs=optim_kwargs,
            features_extractor_class=feat_clazz,
            features_extractor_kwargs=feat_kwargs,
            shared_critic_input_size=critic_input_size,
            net_arch=net_arch,
            activation=policy_cfg.get("activation", "relu"),
            use_popart=policy_cfg.get("use_popart", False),
            device=device,
        )

    return policies


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(exp_cfg: Dict[str, Any], default_device: str) -> Dict[str, Any]:
    device = exp_cfg.get("device", default_device)
    experiment_name = exp_cfg.get("name", "experiment")
    set_global_seed(exp_cfg.get("seed", 42))

    data = build_dataset(exp_cfg["dataset"])
    loaders = build_loaders(data, batch_size=exp_cfg["training"]["batch_size"])

    env_cfg = exp_cfg["env"]
    policy_cfg = exp_cfg["policy"]
    algo_cfg = exp_cfg["algorithm"]
    training_cfg = exp_cfg["training"]

    action_dim = env_cfg["mol_types"]
    obs_dim = env_cfg.get("obs_dim", action_dim)
    if obs_dim != action_dim:
        print("WARNING: obs_dim != mol_types; using mol_types as obs_dim for the current environment API")
        obs_dim = action_dim
    max_mol_out = env_cfg.get("max_mol_out", action_dim)

    encoder, decoder = load_or_init_models(
        dataset_name=exp_cfg["dataset"].get("name", "dataset"),
        input_dim=data["input_dim"],
        obs_dim=obs_dim,
        n_classes=data["n_classes"],
        encoder_hidden=env_cfg.get("encoder_hidden", []),
        decoder_hidden=env_cfg.get("decoder_hidden", []),
        X_train=data["X_train"],
        X_val=data["X_val"],
        y_train=data["y_train"],
        y_val=data["y_val"],
        batch_size=training_cfg["batch_size"],
        pretrain_cfg={
            **exp_cfg.get("pretraining", {}),
            "encoder_activation": env_cfg.get("encoder_activation", "relu"),
            "decoder_activation": env_cfg.get("decoder_activation", "relu"),
            "ae_lr": exp_cfg.get("pretraining", {}).get("ae_lr", policy_cfg.get("critic_lr", 3e-4)),
            "classifier_lr": exp_cfg.get("pretraining", {}).get("classifier_lr", policy_cfg.get("critic_lr", 3e-4)),
        },
        device=device,
    )

    action_space, obs_space = build_spaces(action_dim, obs_dim, env_cfg["n_agents"], max_mol_out)
    if env_cfg.get("space_shape"):
        space_shape = tuple(env_cfg["space_shape"])
    else:
        density = float(env_cfg.get("density", 1.0))
        side = math.sqrt(env_cfg["n_agents"] / density)
        space_shape = (side, side)

    layout_seed = env_cfg.get("seed", 42)
    env_builder = partial(
        build_environment,
        env_cfg=env_cfg,
        action_space=action_space,
        obs_space=obs_space,
        encoder=encoder,
        decoder=decoder,
        X_train=data["X_train"],
        y_train=data["y_train"],
        action_dim=action_dim,
        device=device,
        space_shape=space_shape,
        layout_seed=layout_seed,
    )
    reference_env = env_builder(seed=layout_seed)

    n_partitions = max(1, int(env_cfg.get("partitions", 1)))
    partition_method = env_cfg.get("partition_method", "distance_based")
    partition_mapping = partition_relay_agents(reference_env, n_partitions=n_partitions, method=partition_method)
    agent_policy_mapping = build_agent_policy_mapping(reference_env, partition_mapping)

    if reference_env.shared_obs:
        critic_obs_dim = reference_env.mol_types + reference_env.n_agents * (reference_env.n_agents - 1) // 2
    else:
        critic_obs_dim = obs_dim
    policies = build_policies(policy_cfg, action_space, obs_space, partition_mapping, critic_obs_dim, device)

    if algo_cfg.get("name", "mappo").lower() != "mappo":
        raise ValueError("Only MAPPO is supported in this entry point.")

    algorithm = MAPPO(device=device, **algo_cfg.get("params", {}))
    use_multiprocess_env = bool(env_cfg.get("use_multiprocess", True))
    n_envs = int(env_cfg.get("n_envs", training_cfg.get("batch_size", 1))) if use_multiprocess_env else 1
    n_workers = int(env_cfg.get("n_workers", min(n_envs, 4))) if use_multiprocess_env else 1
    start_method = env_cfg.get("start_method", "fork")
    logging_cfg = exp_cfg.get("logging", {})
    tensorboard_log_dir = None

    if logging_cfg.get("tensorboard", True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = logging_cfg.get("run_name", f"{experiment_name}_{timestamp}")
        tensorboard_log_dir = Path(logging_cfg.get("log_dir", "runs/tensorboard")) / run_name
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        with (tensorboard_log_dir / "config.json").open("w", encoding="utf-8") as fp:
            json.dump(exp_cfg, fp, indent=2)
        print(f"TensorBoard log dir: {tensorboard_log_dir}")

    if use_multiprocess_env:
        env = MultiProcessEnvWrapper(
            env_builder=env_builder,
            n_envs=n_envs,
            n_workers=n_workers,
            seed=layout_seed,
            start_method=start_method,
        )
    else:
        env = reference_env
        print("Debug mode: using raw unwrapped Environment; n_envs and n_workers are forced to 1")

    try:
        learner = MAPPOLearner(
            env=env,
            algorithm=algorithm,
            policies=policies,
            agent_policy_mapping=agent_policy_mapping,
            action_dim=action_dim,
            obs_dim=obs_dim,
            rollout_len=training_cfg["rollout_len"],
            data_batch_size=training_cfg["batch_size"],
            val_loader=loaders["val_loader"],
            test_loader=loaders["test_loader"],
            gamma=training_cfg.get("gamma", 0.99),
            gae_lambda=training_cfg.get("gae_lambda", 0.95),
            critic_obs_dim=critic_obs_dim,
            lr_scheduler_config=exp_cfg.get("lr_scheduler"),
            verbose=exp_cfg.get("verbose", False),
            device=device,
            entropy_scheduler_config=exp_cfg.get("entropy_scheduler"),
            tensorboard_log_dir=tensorboard_log_dir,
            log_interval=logging_cfg.get("log_interval", 10),
        )

        learner.train(epochs=training_cfg["episodes"])

        return {
            "val_acc": learner.metrics.last("val_acc"),
            "test_acc": learner.metrics.last("test_acc"),
        }
    finally:
        env.close()


def list_experiments(config: Dict[str, Any]) -> List[str]:
    return [exp["name"] for exp in config.get("experiments", [])]


def select_experiments(config: Dict[str, Any], requested: List[str] | None) -> List[Dict[str, Any]]:
    experiments = config.get("experiments", [])
    if not requested:
        return experiments
    selected = [exp for exp in experiments if exp.get("name") in requested]
    missing = set(requested) - {exp.get("name") for exp in selected}
    if missing:
        raise ValueError(f"Experiments not found in config: {', '.join(sorted(missing))}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL4BIOCOMM experiments from config")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments.json"),
        help="Path to experiments configuration file",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Names of experiments to run (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.list:
        for name in list_experiments(config):
            print(name)
        return

    to_run = select_experiments(config, args.experiments)
    if not to_run:
        raise ValueError("No experiments found to run.")

    default_device = config.get("default_device", "cpu")

    for exp_cfg in to_run:
        name = exp_cfg.get("name", "experiment")
        print(f"\n=== Running experiment: {name} ===")
        metrics = run_experiment(exp_cfg, default_device)
        print(f"Completed {name} | val_acc={metrics['val_acc']}, test_acc={metrics['test_acc']}")


if __name__ == "__main__":
    main()
