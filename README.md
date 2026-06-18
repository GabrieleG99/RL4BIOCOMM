# RL4BIOCOMM

Master thesis project about emergent molecular communication in bio-inspired reinforcement-learning environments.

The repository trains multi-agent policies for biological communication experiments. The main entry point is
[`main.py`](main.py), which loads experiment definitions from JSON, builds the dataset and environment, initializes
or trains encoder/classifier checkpoints, and runs MAPPO training.

## Repository Layout

- `main.py` - configuration-driven experiment runner.
- `configs/experiments.json` - default experiment definitions.
- `algorithms/` - reinforcement-learning algorithms, including MAPPO.
- `envs/` - biological communication environments and multiprocessing wrappers.
- `learners/` - training loops, rollout collection, metrics, and TensorBoard logging.
- `models/` - neural-network layers, agents, policies, and PopArt support.
- `dists/` - action distribution utilities.
- `utils/` - data loading, schedulers, rollout buffers, and metric logging.
- `testing/` - pytest-based tests.
- `best_models/` - pretrained encoder/classification-head checkpoints used by default experiments.
- `runs/` - generated profiling and training outputs.

## Environment Setup

The project is expected to run in the `RL4BIOCOMM` conda environment.

```bash
conda activate RL4BIOCOMM
```

If the environment does not exist yet, create one from `environment.yml` and then activate it:

```bash
conda env create -f environment.yml
conda activate RL4BIOCOMM
```

Alternatively, install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

TensorBoard logging is enabled in the default config. If `torch.utils.tensorboard` cannot import `SummaryWriter`,
install TensorBoard in the active environment:

```bash
pip install tensorboard
```

## Running Experiments

List the experiments available in the default config:

```bash
python main.py --list
```

Current default experiments:

- `iris_mappo`
- `wine_mappo_shared_obs`

Run all configured experiments:

```bash
python main.py
```

Run one or more named experiments:

```bash
python main.py --experiments iris_mappo
python main.py --experiments iris_mappo wine_mappo_shared_obs
```

Use a custom configuration file:

```bash
python main.py --config path/to/experiments.json
```

Combine a custom config with selected experiments:

```bash
python main.py --config path/to/experiments.json --experiments my_experiment
```

## Configuration

Experiments are defined in `configs/experiments.json` under the top-level `experiments` array. Each experiment contains:

- `name` and `seed` - run identity and reproducibility settings.
- `device` - PyTorch device, usually `cpu` or `cuda`.
- `dataset` - UCI or synthetic classification data settings.
- `env` - agent count, molecule/action dimensions, topology, partitions, multiprocessing, and environment behavior.
- `policy` - actor/critic architecture, feature extractor, learning rates, activation, and PopArt settings.
- `algorithm` - MAPPO hyperparameters.
- `training` - episode count, rollout length, batch size, discount, and GAE lambda.
- `pretraining` - encoder/classification-head checkpoint training and storage settings.
- `logging` - TensorBoard output location and logging interval.
- `lr_scheduler` and `entropy_scheduler` - optional scheduler configs.

Only `mappo` is supported by `main.py` at the moment.

## Data and Checkpoints

For `dataset.source = "uci"`, data is loaded through `ucimlrepo` using the dataset `name`, such as `iris` or `wine`.
For `dataset.source = "synthetic"`, data is generated locally with the configured sample, feature, and class counts.

Before RL training, `main.py` looks for pretrained models in `pretraining.model_save_path`, which defaults to
`best_models/`:

- `best_encoder_<dataset>.pt`
- `best_ch_<dataset>.pt`

If the files are missing or incompatible with the current dimensions, the runner trains a new autoencoder encoder and
classification head, then saves the best checkpoints.

## Outputs

When TensorBoard logging is enabled, each run creates:

```text
runs/tensorboard/<experiment_name>_<timestamp>/
```

The run directory includes a copy of the experiment `config.json` and TensorBoard scalar logs. View logs with:

```bash
tensorboard --logdir runs/tensorboard
```

The training process prints periodic summaries according to `logging.log_interval` and finishes each experiment with
the latest validation and test accuracy.

## Testing

Run the test suite from the repository root:

```bash
pytest -q
```

When using conda without activating the shell first:

```bash
conda run -n RL4BIOCOMM pytest -q
```
