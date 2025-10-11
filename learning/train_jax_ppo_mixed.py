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
from xml.parsers.expat import model

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
# !!
from learning import train_mixed as ppo

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper_mixed as wrapper
from mujoco_playground.config import locomotion_mixed_params

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
_IMPL                      = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
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
_NUM_VIDEOS                = flags.DEFINE_integer("num_videos", 1, "Number of videos to record after training.")
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
_EVAL_ALL_MODELS           = flags.DEFINE_boolean("eval_all_models", False, "Whether to evaluate on all models during training (slower) or just the first model")


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  """Loads the appropriate RL configuration for the environment."""
  if env_name in mujoco_playground.locomotion_mixed._envs:
    if _VISION.value: return locomotion_mixed_params.brax_vision_ppo_config(env_name)
    return locomotion_mixed_params.brax_ppo_config(env_name)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")

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

  # Don't use any unparsed command-line arguments
  del argv

  # Load environment and RL configurations
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

  env = registry.load(_ENV_NAME.value, config=env_cfg)
  
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  print(f"Environment Config:\n{env_cfg}")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # Set up logging and experiment directories
  exp_name = f"{_ENV_NAME.value}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
  if _SUFFIX.value is not None: exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if enabled
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb_config = {"project": _WANDB_PROJECT.value, "name": exp_name}
    if _WANDB_ENTITY.value:
      wandb_config["entity"] = _WANDB_ENTITY.value
    wandb.init(**wandb_config)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})
  
  # Initialize TensorBoard if enabled
  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = SummaryWriter(logdir)
  
  # Handle checkpoint loading and saving
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

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")
  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  # Prepare training function
  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = ppo_networks_vision.make_ppo_networks_vision if _VISION.value else ppo_networks.make_ppo_networks
  network_factory = functools.partial(network_fn, **ppo_params.network_factory) if hasattr(ppo_params, "network_factory") else network_fn

  # if _DOMAIN_RANDOMIZATION.value:
  #   training_params["randomization_fn"] = registry.get_domain_randomizer(_ENV_NAME.value)
  # !! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  partial_wrap_fn = functools.partial(
    wrapper.wrap_for_mixed_brax_training,
    model_cfg=env_cfg.models_config
  )
  # !! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  if _VISION.value:
    env = wrapper.wrap_for_mixed_brax_training(
    # ! env = wrapper.wrap_for_brax_training(
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
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=None if _VISION.value else partial_wrap_fn, # ! wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )

  times = [time.monotonic()]

  # Define a progress function for logging metrics
  def progress(num_steps, metrics):
    times.append(time.monotonic())
    
    # Prepare enhanced metrics for logging
    enhanced_metrics = dict(metrics)
    
    # !!
    # Add detailed model distribution and reward analysis
    if "model_distribution/total_envs" in metrics:
      total_envs = int(metrics["model_distribution/total_envs"])
      for i in range(5):  # Check up to 5 models
        count_key = f"model_distribution/model_{i}_count"
        percent_key = f"model_distribution/model_{i}_percentage"
        # if count_key in metrics and percent_key in metrics:
        #   enhanced_metrics[f"model_dist/model_{i}_count"] = metrics[count_key]
        #   enhanced_metrics[f"model_dist/model_{i}_percent"] = metrics[percent_key]
    
    # Add reward component analysis
    for key, value in metrics.items():
      if key.startswith('reward/') and not key.endswith('_std'):
        enhanced_metrics[f"reward_components/{key.replace('reward/', '')}"] = value
    
    if _USE_WANDB.value and not _PLAY_ONLY.value and wandb is not None:
      wandb.log(enhanced_metrics, step=num_steps)
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in enhanced_metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value and 'eval/episode_reward' in metrics:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )
      # # !!
      # # Print model distribution if available
      # if "model_distribution/total_envs" in metrics:
      #   total_envs = int(metrics["model_distribution/total_envs"])
      #   print(f"{num_steps}: total envs={total_envs}")
      #   for i in range(5):  # Check up to 5 models
      #     count_key = f"model_distribution/model_{i}_count"
      #     percent_key = f"model_distribution/model_{i}_percentage"
      #     if count_key in metrics and percent_key in metrics:
      #       count = int(metrics[count_key])
      #       percent = metrics[percent_key]
      #       print(f"  Model {i}: {count} envs ({percent:.1f}%)")
      
  
  # Load evaluation environments for all models
  config_overrides = {"impl": _IMPL.value}
  if _VISION.value:
    eval_envs = None
  else:
    # Import the constants to get the MuJoCo models
    from mujoco_playground._src.locomotion_mixed.kawaru import kawaru_constants as consts
    
    # Create evaluation environments for all models used during training
    num_models = len(env_cfg.models_config.xml_paths)
    eval_envs = []
    for model_idx in range(num_models):
      # Pass the correct MuJoCo model for rendering
      eval_env = registry.load(_ENV_NAME.value, config=env_cfg, config_overrides=config_overrides, 
                             add_model_info=True, model_index=model_idx, 
                             mj_model=consts.MJ_MODEL[model_idx])
      eval_envs.append(eval_env)
    print(f"Created {num_models} evaluation environments for models 0-{num_models-1}")
  # ! num_envs = 1
  # ! if _VISION.value:
  # !   num_envs = env_cfg.vision_config.render_batch_size

  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    from rscope import brax as rscope_utils
    
    rscope_env =  env if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg, config_overrides=config_overrides)
    
    if not _VISION.value:
      rscope_env = wrapper.wrap_for_mixed_brax_training(
      # ! rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

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

  # Train the agent (evaluation during training is feedforward only - doesn't modify policy)
  if _EVAL_ALL_MODELS.value and eval_envs:
    # Use all evaluation environments during training (slower but more comprehensive)
    training_eval_env = eval_envs
    print(f"Training will evaluate on all {len(eval_envs)} models during training")
  else:
    # Use only first evaluation environment during training (faster, recommended)
    training_eval_env = eval_envs[0] if eval_envs else None
    if eval_envs:
      print(f"Training will evaluate only on model 0 during training (use --eval_all_models=true for all models)")
  
  make_inference_fn, params, _ = train_fn(
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=training_eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]:.2f}s")
    print(f"Time to train: {times[-1] - times[1]:.2f}s")

  # Start inference and evaluation
  print("Starting inference...")

  # Create inference function.
  inference_fn = make_inference_fn(params, deterministic=True)  
  jit_inference_fn = jax.jit(inference_fn)

  # Run evaluation rollouts for all models.
  def do_rollout(rng, state, eval_env):
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})  # pytype: disable=attribute-error
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_key)[0]
      state = eval_env.step(state, act)
      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.time": state.data.time,
          "data.ctrl": state.data.ctrl,
          "data.mocap_pos": state.data.mocap_pos,
          "data.mocap_quat": state.data.mocap_quat,
          "data.xfrc_applied": state.data.xfrc_applied,
          # !! to draw command 
          "data": state.data, # <-- This saves all data fields, including xpos and xmat
          "info": {"command": state.info["command"]}, 
      })
      if _VISION.value:
        traj_data = jax.tree_util.tree_map(lambda x: x[0], traj_data)
      return (state, rng), traj_data

    _, traj = jax.lax.scan(
        step, (state, rng), None, length=_EPISODE_LENGTH.value
    )
    return traj

  # Generate rollouts for each model using the same RNG for fair comparison
  all_trajectories = []
  if eval_envs:
    # Use the same RNG seed for all models to ensure identical initial conditions
    base_rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
    
    for model_idx, eval_env in enumerate(eval_envs):
      print(f"Generating rollouts for model {model_idx}...")
      # Use the same RNG for all models to ensure fair comparison
      rng = base_rng
      reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)
      if _VISION.value:
        reset_states = jax.tree_util.tree_map(lambda x: x[0], reset_states)
      
      # Create a partial function with the eval_env bound
      def do_rollout_model(rng, state):
        return do_rollout(rng, state, eval_env)
      
      traj_stacked = jax.jit(jax.vmap(do_rollout_model))(rng, reset_states)
      
      model_trajectories = [None] * _NUM_VIDEOS.value
      for i in range(_NUM_VIDEOS.value):
        t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
        model_trajectories[i] = [
            jax.tree.map(lambda x, j=j: x[j], t)
            for j in range(_EPISODE_LENGTH.value)
        ]
      all_trajectories.append((model_idx, eval_env, model_trajectories))
  else:
    # Fallback for vision case - use the same base RNG
    eval_env = training_eval_env
    base_rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
    rng = base_rng
    reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)
    if _VISION.value:
      reset_states = jax.tree_util.tree_map(lambda x: x[0], reset_states)
    
    def do_rollout_model(rng, state):
      return do_rollout(rng, state, eval_env)
    
    traj_stacked = jax.jit(jax.vmap(do_rollout_model))(rng, reset_states)
    
    trajectories = [None] * _NUM_VIDEOS.value
    for i in range(_NUM_VIDEOS.value):
      t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
      trajectories[i] = [
          jax.tree.map(lambda x, j=j: x[j], t)
          for j in range(_EPISODE_LENGTH.value)
      ]
    all_trajectories.append((0, eval_env, trajectories))

  # Render and save rollouts for all models.
  render_every = 2
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  
  # Import the draw function for joystick commands
  import numpy as np
  from mujoco_playground._src.gait import draw_joystick_command
  
  for model_idx, eval_env, trajectories in all_trajectories:
    fps = 1.0 / eval_env.dt / render_every
    print(f"Rendering videos for model {model_idx} (FPS: {fps:.1f})...")
    
    for i, rollout in enumerate(trajectories):
      traj = rollout[::render_every]
      
      # Create modify_scene_fns for this specific rollout
      modify_scene_fns = []
      for state in rollout:
          # Il comando è stato salvato in info["command"] durante lax.scan
          command = np.array(state.info["command"]) 
          
          # Estrazione degli altri dati
          xyz = np.array(state.data.xpos[eval_env._torso_body_id])
          xyz += np.array([0, 0, 0.2])
          x_axis = np.array(state.data.xmat[eval_env._torso_body_id, 0])
          yaw = -np.arctan2(x_axis[1], x_axis[0])
          
          modify_scene_fns.append(
              functools.partial(
                  draw_joystick_command,
                  cmd=command, # Passa il comando dinamico salvato
                  xyz=xyz,
                  theta=yaw,
                  scl=abs(command[0]) / env_cfg.command_config.a[0],
              )
          )
      
      mod_fns = modify_scene_fns[::render_every]
      assert len(traj) == len(mod_fns)
      frames = eval_env.render(
          traj, 
          height=480, 
          width=640, 
          modify_scene_fns=mod_fns,
          scene_option=scene_option, 
          camera="track",
      )
      video_filename = f"rollout_model{model_idx}_env{i}.mp4"
      media.write_video(video_filename, frames, fps=fps)
      print(f"Rollout video saved as '{video_filename}'.")


if __name__ == "__main__":
  app.run(main)