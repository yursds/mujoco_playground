# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment."""

import functools
import json
import os
import time
import warnings
from datetime import datetime
from typing import Dict, Any

from absl import app, flags, logging
import jax
import jax.numpy as jp
import mediapy as media
from etils import epath
from ml_collections import config_dict
import mujoco
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter
import wandb

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_mixed_params
from learning import hrl_train as ppo
from learning import multi_rscope as rscope_utils

# Set up environment variables for JAX and rendering
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Suppress log and warning messages from JAX and its dependencies
logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

# Define command-line flags for configuration
_ACTION_REPEAT             = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_BATCH_SIZE                = flags.DEFINE_integer("batch_size", 256, "Batch size")
_CLIPPING_EPSILON          = flags.DEFINE_float("clipping_epsilon", 0.2, "Clipping epsilon for PPO")
_DETERMINISTIC_RSCOPE      = flags.DEFINE_boolean("deterministic_rscope", True, "Run deterministic rollouts for rscope viewer")
_DISCOUNTING               = flags.DEFINE_float("discounting", 0.97, "Discounting factor")
_DOMAIN_RANDOMIZATION      = flags.DEFINE_boolean("domain_randomization", False, "Use domain randomization")
_ENV_NAME                  = flags.DEFINE_string("env_name", "LeapCubeReorient", f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}")
_ENTROPY_COST              = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_EPISODE_LENGTH            = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_LEARNING_RATE             = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_LOAD_CHECKPOINT_PATH      = flags.DEFINE_string("load_checkpoint_path", None, "Path to load a checkpoint from")
_LOG_TRAINING_METRICS      = flags.DEFINE_boolean("log_training_metrics", False, "Log training metrics")
_NORMALIZE_OBSERVATIONS    = flags.DEFINE_boolean("normalize_observations", True, "Normalize observations")
_NUM_TIMESTEPS             = flags.DEFINE_integer("num_timesteps", 1_000_000, "Number of timesteps")
_NUM_EVALS                 = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_NUM_EVAL_ENVS             = flags.DEFINE_integer("num_eval_envs", 128, "Number of evaluation environments")
_NUM_ENVS                  = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_MINIBATCHES           = flags.DEFINE_integer("num_minibatches", 8, "Number of minibatches")
_NUM_UPDATES_PER_BATCH     = flags.DEFINE_integer("num_updates_per_batch", 8, "Number of updates per batch")
_MAX_GRAD_NORM             = flags.DEFINE_float("max_grad_norm", 1.0, "Max gradient norm")
_PLAY_ONLY                 = flags.DEFINE_boolean("play_only", False, "If true, only play with the model and do not train")
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list("policy_hidden_layer_sizes", [64, 64, 64], "Policy network hidden layer sizes")
_POLICY_OBS_KEY            = flags.DEFINE_string("policy_obs_key", "state", "Observation key for the policy network")
_REWARD_SCALING            = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_RSCOPE_ENVS               = flags.DEFINE_integer("rscope_envs", None, "Number of parallel environment rollouts to save for the rscope viewer")
_RUN_EVALS                 = flags.DEFINE_boolean("run_evals", True, "Run evaluation rollouts between policy updates")
_SEED                      = flags.DEFINE_integer("seed", 1, "Random seed")
_SUFFIX                    = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_TRAINING_METRICS_STEPS    = flags.DEFINE_integer("training_metrics_steps", 1_000_000, "Number of steps between logging training metrics")
_UNROLL_LENGTH             = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_USE_TB                    = flags.DEFINE_boolean("use_tb", False, "Use TensorBoard for logging")
_USE_WANDB                 = flags.DEFINE_boolean("use_wandb", True, "Use Weights & Biases for logging")
_VALUE_HIDDEN_LAYER_SIZES  = flags.DEFINE_list("value_hidden_layer_sizes", [64, 64, 64], "Value network hidden layer sizes")
_VALUE_OBS_KEY             = flags.DEFINE_string("value_obs_key", "state", "Observation key for the value network")
_VISION                    = flags.DEFINE_boolean("vision", False, "Use vision input")
_WANDB_PROJECT             = flags.DEFINE_string("wandb_project", "mujoco-playground-rl", "Weights & Biases project name")
_WANDB_ENTITY              = flags.DEFINE_string("wandb_entity", None, "Weights & Biases entity name")

def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  """Loads the appropriate RL configuration for the environment."""
  if env_name in mujoco_playground.locomotion_mixed._envs:
    if _VISION.value: return locomotion_mixed_params.brax_vision_ppo_config(env_name)
    return locomotion_mixed_params.brax_ppo_config(env_name)

def setup_experiment_logging(
  exp_name: str, 
  env_cfg: config_dict.ConfigDict, 
  logdir: epath.Path
  ) -> SummaryWriter | None:
  
  """Sets up experiment directories and initializes logging tools."""

  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb_config = {"project": _WANDB_PROJECT.value, "name": exp_name}
    if _WANDB_ENTITY.value:
      wandb_config["entity"] = _WANDB_ENTITY.value
    wandb.init(**wandb_config)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = SummaryWriter(logdir)
    return writer
  return None

def get_checkpoint_path(env_cfg: config_dict.ConfigDict, logdir: epath.Path) -> epath.Path:
  """Determines the path for loading or saving checkpoints."""
  
  # Handle checkpoint loading and saving
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  restore_checkpoint_path = None
  if _LOAD_CHECKPOINT_PATH.value is not None:
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      # latest_ckpts = sorted([c for c in ckpt_path_abs.glob("*") if c.is_dir()], key=lambda x: int(x.name))
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
    print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None
  
  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  return restore_checkpoint_path

def configure_training(
  ppo_params: config_dict.ConfigDict, 
  env_cfg: config_dict.ConfigDict, 
  exp_name: str, 
  logdir: epath.Path
  ) -> tuple[functools.partial, Any, Any]:
  
  """Configures and returns the training function, environment, and evaluation environment."""
  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = ppo_networks_vision.make_ppo_networks_vision if _VISION.value else ppo_networks.make_ppo_networks
  network_factory = functools.partial(network_fn, **ppo_params.network_factory) if hasattr(ppo_params, "network_factory") else network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(_ENV_NAME.value)

  if _VISION.value:
    env = wrapper.wrap_for_brax_training(
        env,
        vision=True,
        num_vision_envs=env_cfg.vision_config.render_batch_size,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.action_repeat,
        randomization_fn=training_params.get("randomization_fn"),
    )

  num_eval_envs = ppo_params.num_envs if _VISION.value else ppo_params.get("num_eval_envs", 128)
  if "num_eval_envs" in training_params: del training_params["num_eval_envs"]

  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=get_checkpoint_path(env_cfg, logdir),
      save_checkpoint_path=epath.Path("logs").resolve() / exp_name / "checkpoints",
      wrap_env_fn=None if _VISION.value else wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )

  eval_env = None if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg)
  return train_fn, eval_env, training_params

def run_evaluation_and_render(make_inference_fn, params, env_cfg):
  """Runs a single evaluation rollout and saves a video."""
  print("Starting inference...")
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  eval_env = (None if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg))
  num_envs = 1
  if _VISION.value:
    num_envs = env_cfg.vision_config.render_batch_size

  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(123)
  rng, reset_rng = jax.random.split(rng)
  if _VISION.value:
    reset_rng = jp.asarray(jax.random.split(reset_rng, num_envs))

  state = jit_reset(reset_rng)
  state0 = jax.tree_util.tree_map(lambda x: x[0], state) if _VISION.value else state
  rollout = [state0]

  for _ in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    state0 = jax.tree_util.tree_map(lambda x: x[0], state) if _VISION.value else state
    rollout.append(state0)
    if state0.done:
      break

  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  print(f"FPS for rendering: {fps}")

  traj = rollout[::render_every]
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  frames = eval_env.render(
      traj,
      camera="track",
      scene_option=scene_option,
      width=640,
      height=480,
  )
  media.write_video("rollout.mp4", frames, fps=fps)
  print("Rollout video saved as 'rollout.mp4'.")

def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  
  :param full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  :param obs: nd.array or dict obs based on env configuration
  :param rew: nd.array rewards
  :param done: nd.array done flags
  """
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )

def main(argv):
  """Main function to run training and evaluation."""
  del argv

  # Load configurations
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  ppo_params = get_rl_config(_ENV_NAME.value)
  
  # Override configurations with command-line flags
  if _NUM_TIMESTEPS.present: ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present: ppo_params.num_timesteps = 0
  if _NUM_EVALS.present: ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present: ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present: ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present: ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present: ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present: ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present: ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present: ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present: ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present: ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present: ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present: ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present: ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present: ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present: ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present: ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present: ppo_params.network_factory.policy_hidden_layer_sizes = list(map(int, _POLICY_HIDDEN_LAYER_SIZES.value))
  if _VALUE_HIDDEN_LAYER_SIZES.present: ppo_params.network_factory.value_hidden_layer_sizes = list(map(int, _VALUE_HIDDEN_LAYER_SIZES.value))
  if _POLICY_OBS_KEY.present: ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present: ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value
  if _VISION.value:
    env_cfg.vision = True
    env_cfg.vision_config.render_batch_size = ppo_params.num_envs
  if _RUN_EVALS.present: ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present: ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present: ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  # Set up experiment logging
  exp_name = f"{_ENV_NAME.value}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
  if _SUFFIX.value is not None: exp_name += f"-{_SUFFIX.value}"
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  writer = setup_experiment_logging(exp_name, env_cfg, logdir)

  # Load initial environment
  env = registry.load(_ENV_NAME.value, config=env_cfg)

  # Configure and run training
  train_fn, eval_env, training_params = configure_training(ppo_params, env_cfg, exp_name, logdir)

  """Executes the main training loop and handles progress logging."""
  times = [time.monotonic()]
  def progress(num_steps: int, metrics: Dict[str, Any]):
    times.append(time.monotonic())
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(f"{num_steps}: mean episode reward={metrics['episode/sum_reward']:.3f}")

  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    rscope_env = env if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg)
    if not _VISION.value:
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        ppo_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )
    def policy_params_fn(current_step, make_policy, params):
      rscope_handle.set_make_policy(make_policy)
      rscope_handle.dump_rollout(params)

  make_inference_fn, params, _ = train_fn(
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]:.2f}s")
    print(f"Time to train: {times[-1] - times[1]:.2f}s")
  
  # Run evaluation and rendering
  run_evaluation_and_render(make_inference_fn, params, env_cfg)

if __name__ == "__main__":
  app.run(main)