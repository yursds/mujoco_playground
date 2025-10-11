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
"""Wrappers for MuJoCo Playground environments."""

from typing import Any, Callable, List, Optional, Dict, Union, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp
from mujoco import mjx, MjModel
from ml_collections import config_dict

from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper, BraxAutoResetWrapper



def wrap_for_mixed_brax_training(
    env: mjx_env.MjxEnv,
    model_cfg: config_dict.ConfigDict,
    episode_length: int = 1000,
    action_repeat: int = 1,
    full_reset: bool = False,
    randomization_fn: Optional[Callable[[mjx.Model], mjx.Model]] = None,
    include_model_info_in_obs: bool = True,
    use_fast_wrapper: bool = False,
) -> Wrapper:
  """Factory that returns MixedEnvWrapper wrapped with Episode & AutoReset.
  
  Args:
    env: Base environment
    model_cfg: Model configuration with xml_paths
    episode_length: Length of episodes
    action_repeat: Action repeat factor
    full_reset: Whether to do full resets
    randomization_fn: Optional randomization function
    include_model_info_in_obs: Whether to include model info in observations
    use_fast_wrapper: Whether to use the fast JAX-optimized wrapper
  """
  config = model_cfg.config if hasattr(model_cfg, "config") else getattr(env, "_config", model_cfg)
  
  wrapper_class = MixedEnvWrapperFast if use_fast_wrapper else MixedEnvWrapper
  env = wrapper_class(
      env, 
      model_cfg.xml_paths, 
      config, 
      randomization_fn=randomization_fn,
      include_model_info_in_obs=include_model_info_in_obs
  )
  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper(env, full_reset=full_reset)
  return env


def wrap_for_mixed_brax_training_fast(
    env: mjx_env.MjxEnv,
    model_cfg: config_dict.ConfigDict,
    episode_length: int = 1000,
    action_repeat: int = 1,
    full_reset: bool = False,
    randomization_fn: Optional[Callable[[mjx.Model], mjx.Model]] = None,
    include_model_info_in_obs: bool = True,
) -> Wrapper:
  """Factory that returns MixedEnvWrapperFast wrapped with Episode & AutoReset."""
  return wrap_for_mixed_brax_training(
      env=env,
      model_cfg=model_cfg,
      episode_length=episode_length,
      action_repeat=action_repeat,
      full_reset=full_reset,
      randomization_fn=randomization_fn,
      include_model_info_in_obs=include_model_info_in_obs,
      use_fast_wrapper=True
  )


class MixedEnvWrapperFast(Wrapper):
  """JAX-optimized wrapper that manages multiple MuJoCo models efficiently.
  
  This version processes all models in parallel and uses vectorized operations
  for better computational efficiency compared to MixedEnvWrapper.
  
  Actions can include a tail for model selection: if action size equals 
  base_action_size + num_models, the tail is treated as one-hot selection.
  """

  def __init__(
    self,
    dummy_env: mjx_env.MjxEnv,
    xml_paths: List[str],
    config: config_dict.ConfigDict,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    randomization_fn: Optional[Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]] = None,
    include_model_info_in_obs: bool = True,
  ):
    super().__init__(dummy_env)
    
    self.num_models = len(xml_paths)
    if self.num_models < 2:
      raise ValueError(f"MixedEnvWrapperFast requires at least 2 models, got {self.num_models}")

    self.include_model_info_in_obs = include_model_info_in_obs
    
    # Create environments for each model
    self.environments = [
        dummy_env.__class__(mj_model=MjModel.from_xml_path(path), config=config, config_overrides=config_overrides)
        for path in xml_paths
    ]
    
    # Pre-compile vectorized functions for all models
    self._vmap_reset_all = jax.vmap(lambda env, key: env.reset(key), in_axes=(None, 0))
    self._vmap_step_all = jax.vmap(lambda env, state, action: env.step(state, action), in_axes=(None, 0, 0))
    
    self._pending_selection = None

  def _reset_all_models(self, rng_keys: jax.Array) -> List[mjx_env.State]:
    """Reset all models in parallel and return list of states."""
    # Reset each model with its corresponding key
    all_states = []
    for i, env in enumerate(self.environments):
      state = env.reset(rng_keys[i])
      all_states.append(state)
    return all_states

  def _step_all_models(self, states: List[mjx_env.State], actions: jax.Array) -> List[mjx_env.State]:
    """Step all models in parallel with their corresponding states and actions."""
    all_new_states = []
    for i, env in enumerate(self.environments):
      new_state = env.step(states[i], actions)
      all_new_states.append(new_state)
    return all_new_states

  def _select_states_by_model_index(self, all_states: List[mjx_env.State], model_indices: jax.Array, batch_size: int) -> mjx_env.State:
    """Efficiently select states based on model indices using vectorized operations."""
    # Stack all states into arrays for each field
    stacked_data = {}
    stacked_obs = {}
    stacked_reward = jp.stack([state.reward for state in all_states])  # (num_models, batch_size)
    stacked_done = jp.stack([state.done for state in all_states])  # (num_models, batch_size)
    stacked_metrics = {}
    
    # Stack data fields
    first_state = all_states[0]
    for field_name in first_state.data.__annotations__:
      field_values = []
      for state in all_states:
        field_val = getattr(state.data, field_name)
        if field_val is not None:
          field_values.append(field_val)
        else:
          # Handle None values by creating zeros with appropriate shape
          if len(field_values) > 0:
            field_values.append(jp.zeros_like(field_values[0]))
          else:
            field_values.append(None)
      
      if field_values[0] is not None:
        stacked_data[field_name] = jp.stack(field_values)  # (num_models, batch_size, ...)
    
    # Stack observation fields
    for obs_key in first_state.obs:
      obs_values = [state.obs[obs_key] for state in all_states]
      stacked_obs[obs_key] = jp.stack(obs_values)  # (num_models, batch_size, ...)
    
    # Stack metrics
    if first_state.metrics:
      for metric_key in first_state.metrics:
        metric_values = [state.metrics[metric_key] for state in all_states]
        stacked_metrics[metric_key] = jp.stack(metric_values)  # (num_models, batch_size, ...)
    
    # Use advanced indexing to select the right state for each environment
    # model_indices has shape (batch_size,), we want to select stacked_data[model_indices[i], i] for each i
    batch_indices = jp.arange(batch_size)  # [0, 1, 2, ..., batch_size-1]
    
    # Select data
    selected_data = {}
    for field_name, stacked_field in stacked_data.items():
      if stacked_field is not None:
        selected_data[field_name] = stacked_field[model_indices, batch_indices]  # (batch_size, ...)
      else:
        selected_data[field_name] = None
    
    # Select observations
    selected_obs = {}
    for obs_key, stacked_obs_val in stacked_obs.items():
      selected_obs[obs_key] = stacked_obs_val[model_indices, batch_indices]  # (batch_size, ...)
    
    # Select rewards and done flags
    selected_reward = stacked_reward[model_indices, batch_indices]  # (batch_size,)
    selected_done = stacked_done[model_indices, batch_indices]  # (batch_size,)
    
    # Select metrics
    selected_metrics = {}
    for metric_key, stacked_metric in stacked_metrics.items():
      selected_metrics[metric_key] = stacked_metric[model_indices, batch_indices]  # (batch_size,)
    
    # Reconstruct the state
    selected_data_obj = first_state.data.__class__(**selected_data)
    return first_state.replace(
        data=selected_data_obj,
        obs=selected_obs,
        reward=selected_reward,
        done=selected_done,
        metrics=selected_metrics
    )

  def reset(self, rng_key: jax.Array) -> mjx_env.State:
    """Reset environment, consuming pending selection if available."""
    sel = self._pending_selection
    initial_states = self.reset_with_selection(rng_key, pending_selection=sel)
    self._pending_selection = None
    return initial_states

  def reset_with_selection(self, rng_key: jax.Array, pending_selection: Optional[jax.Array] = None) -> mjx_env.State:
    """Fast reset with model selection using parallel processing."""
    batch_size = rng_key.shape[0]
    
    # Determine model indices
    if (pending_selection is not None and 
        hasattr(pending_selection, 'info') and 
        'model_indices_pending' in pending_selection.info):
      pending_indices = pending_selection.info['model_indices_pending']
      use_pending = pending_indices != -1
      
      selection_key = jax.random.split(rng_key[0])[0]
      random_indices = jax.random.randint(
          selection_key, shape=(batch_size,), minval=0, maxval=self.num_models, dtype=int
      )
      
      model_indices = jp.where(use_pending, pending_indices, random_indices)
    else:
      selection_key = jax.random.split(rng_key[0])[0]
      model_indices = jax.random.randint(
          selection_key, shape=(batch_size,), minval=0, maxval=self.num_models, dtype=int
      )

    # Generate separate RNG keys for each model
    model_keys = jax.random.split(jax.random.split(rng_key[0])[1], self.num_models)
    expanded_keys = jp.broadcast_to(rng_key[None, :, :], (self.num_models, batch_size, rng_key.shape[-1]))
    
    # Reset all models in parallel
    all_states = self._reset_all_models(model_keys)
    
    # For each model, we need the full batch, but we'll select the appropriate ones later
    # Expand each model's reset to handle the full batch
    expanded_all_states = []
    for i, env in enumerate(self.environments):
      # Reset this model for all batch positions
      batch_state = env.reset(expanded_keys[i])
      expanded_all_states.append(batch_state)
    
    # Select the appropriate state for each environment based on model_indices
    initial_states = self._select_states_by_model_index(expanded_all_states, model_indices, batch_size)
    
    # Initialize state info
    self._init_state_info(initial_states, batch_size, model_indices)
    
    # Add model information to observations if requested
    if self.include_model_info_in_obs:
      initial_states = self._add_model_info_to_obs(initial_states)
    
    return initial_states

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Fast step using parallel processing of all models."""
    batch_size = action.shape[0]
    base_action = action[..., :self.ctrl_size]
    tail = action[..., self.ctrl_size:]
    
    current_model_indices = state.info['active_model']
    
    # Create states for each model by grouping environments by their current model
    model_states = []
    model_actions = []
    
    for model_idx in range(self.num_models):
      # Find which environments are currently using this model
      mask = current_model_indices == model_idx  # (batch_size,)
      
      # Create a state for this model with only the relevant environments
      # For efficiency, we'll process all environments but mask later
      model_states.append(state)
      model_actions.append(base_action)
    
    # Step all models in parallel
    all_new_states = []
    for i, env in enumerate(self.environments):
      new_state = env.step(model_states[i], model_actions[i])
      all_new_states.append(new_state)
    
    # Select the appropriate new state for each environment
    new_state = self._select_states_by_model_index(all_new_states, current_model_indices, batch_size)
    
    # Preserve and update model information
    new_state.info['active_model'] = current_model_indices
    
    # Parse tail for pending model selection
    if tail.shape[-1] == self.num_models:
      pending_model_indices = jp.argmax(tail, axis=-1).astype(jp.int32)
      new_state.info['model_indices_pending'] = pending_model_indices
    else:
      new_state.info['model_indices_pending'] = state.info['model_indices_pending']
    
    # Add model information to observations if requested
    if self.include_model_info_in_obs:
      new_state = self._add_model_info_to_obs(new_state)
  
    return new_state

  def render(
      self,
      trajectory: List[mjx_env.State],
      height: int = 240,
      width: int = 320,
      camera: Optional[str] = None,
      scene_option=None,
      modify_scene_fns=None,
  ):
    """Render trajectory using the correct model for each frame."""
    frames = []
    
    for state in trajectory:
      model_idx = int(state.info['active_model'])
      
      # Use the appropriate environment to render this frame
      frame = self.environments[model_idx].render(
          [state],
          height=height,
          width=width,
          camera=camera,
          scene_option=scene_option,
          modify_scene_fns=modify_scene_fns,
      )
      frames.append(frame[0])  # Extract single frame from list
    
    return frames

  def _init_state_info(self, state: mjx_env.State, batch_size: int, model_indices: jax.Array):
    """Initialize state info with consistent pytree structure."""
    # Ensure model_indices has the correct batch dimension
    if model_indices.shape == ():
      # Scalar case - expand to (batch_size,)
      model_indices = jp.full((batch_size,), model_indices, dtype=jp.int32)
    elif len(model_indices.shape) == 0:
      # Another scalar case
      model_indices = jp.array([model_indices] * batch_size, dtype=jp.int32)
    
    state.info['active_model'] = model_indices
    state.info['model_indices_pending'] = jp.full((batch_size,), -1, dtype=jp.int32)
    
  def _add_model_info_to_obs(self, state: mjx_env.State) -> mjx_env.State:
    """Add model information to observations efficiently."""
    model_indices = state.info['active_model']  # Shape: (batch_size,)
    model_onehot = jax.nn.one_hot(model_indices, self.num_models, dtype=jp.float32)  # Shape: (batch_size, num_models)
    
    new_obs = {}
    for obs_key, obs_val in state.obs.items():
      if obs_key in ['state', 'privileged_state']:
        enhanced_obs = jp.concatenate([obs_val, model_onehot], axis=-1)
        new_obs[obs_key] = enhanced_obs
      else:
        new_obs[obs_key] = obs_val
    
    return state.replace(obs=new_obs)
  
  @property
  def action_size(self) -> int:
    """Return the effective action size including the selection tail."""
    return self.env.action_size + self.num_models
  
  @property
  def ctrl_size(self) -> int:
    """Return the control size."""
    return self.env.action_size


class MixedEnvWrapper(Wrapper):
  """Wrapper that manages multiple MuJoCo models in a single batched environment.
  
  Actions can include a tail for model selection: if action size equals 
  base_action_size + num_models, the tail is treated as one-hot selection.
  """

  def __init__(
    self,
    dummy_env: mjx_env.MjxEnv,
    xml_paths: List[str],
    config: config_dict.ConfigDict,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    randomization_fn: Optional[Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]] = None,
    include_model_info_in_obs: bool = False,
  ):
    super().__init__(dummy_env)
    
    self.num_models = len(xml_paths)
    if self.num_models < 2:
      raise ValueError(f"MixedEnvWrapper requires at least 2 models, got {self.num_models}")

    self.include_model_info_in_obs = include_model_info_in_obs
    
    # Create environments for each model
    self.environments = [
        dummy_env.__class__(mj_model=MjModel.from_xml_path(path), config=config, config_overrides=config_overrides)
        for path in xml_paths
    ]
    self.reset_fns = tuple(env.reset for env in self.environments)
    self.step_fns = tuple(env.step for env in self.environments)
    self._pending_selection = None

  def reset(self, rng_key: jax.Array) -> mjx_env.State:
    """Reset environment, consuming pending selection if available."""
    sel = self._pending_selection
    initial_states = self.reset_with_selection(rng_key, pending_selection=sel)
    self._pending_selection = None
    return initial_states

  def reset_with_selection(self, rng_key: jax.Array, pending_selection: Optional[jax.Array] = None) -> mjx_env.State:
    """Reset with model selection based on pending indices from policy's tail output.
    
    This method consumes the pending model selection that was set during the previous
    episode's steps via the action tail. This ensures that:
    1. The policy can influence which model to use for the NEXT episode
    2. The model selection is frozen during each rollout
    3. Model changes only happen at episode boundaries (resets)
    """
    
    def _reset_single(rng_key: jax.Array, model_idx: int) -> mjx_env.State:
      return jax.lax.switch(model_idx, self.reset_fns, rng_key)

    batch_size = rng_key.shape[0]
    
    # Check if we have next model selection from previous episode's action tails
    if (pending_selection is not None and 
        hasattr(pending_selection, 'info') and 
        'next_model' in pending_selection.info):
      next_models = pending_selection.info['next_model']
      # Use next model selection if valid (not -1), otherwise fall back to random
      use_next = next_models != -1
      
      # Random selection as fallback for environments without valid next model selection
      selection_key = jax.random.split(rng_key[0])[0]
      random_indices = jax.random.randint(
          selection_key, shape=(batch_size,), minval=0, maxval=self.num_models, dtype=int
      )
      
      # Use next model selection where available, random otherwise
      model_indices = jp.where(use_next, next_models, random_indices)
    else:
      # No pending selection available, use uniform random selection for all environments
      selection_key = jax.random.split(rng_key[0])[0]
      model_indices = jax.random.randint(
          selection_key, shape=(batch_size,), minval=0, maxval=self.num_models, dtype=int
      )

    # Reset each environment with its selected model
    initial_states = jax.vmap(_reset_single)(rng_key, model_indices)
    self._init_state_info(initial_states, batch_size, model_indices)
    
    # Add model information to observations if requested (allows policy to condition on current model)
    if self.include_model_info_in_obs:
      initial_states = self._add_model_info_to_obs(initial_states)
    
    return initial_states
  
  def set_pending_selection(self, pending_state: mjx_env.State) -> None:
    """Set pending selection for next reset (useful for debugging/testing)."""
    self._pending_selection = pending_state
  
  def _parse_selection(self, pending_selection: Optional[jax.Array], batch_size: int) -> Optional[jax.Array]:
    """Parse pending selection into model indices."""
    if pending_selection is None:
      return None
    
    sel = jp.asarray(pending_selection)
    if len(sel.shape) == 2 and sel.shape == (batch_size, self.num_models):
      return jp.argmax(sel, axis=-1).astype(int)
    elif len(sel.shape) >= 1 and sel.shape[0] == batch_size:
      return sel.astype(int)
    return None
  
  def _init_state_info(self, state: mjx_env.State, batch_size: int, model_indices: jax.Array):
    """Initialize state info with consistent pytree structure."""
    # Ensure model_indices has the correct batch dimension
    if model_indices.shape == ():
      # Scalar case - expand to (batch_size,)
      model_indices = jp.full((batch_size,), model_indices, dtype=jp.int32)
    elif len(model_indices.shape) == 0:
      # Another scalar case
      model_indices = jp.array([model_indices] * batch_size, dtype=jp.int32)
    
    # Currently active model for each environment
    state.info['active_model'] = model_indices
    # Model to use for next reset (initialized to -1 = no selection)
    state.info['next_model'] = jp.full((batch_size,), -1, dtype=jp.int32)
    
  def _add_model_info_to_obs(self, state: mjx_env.State) -> mjx_env.State:
    """Add model information to observations.
    
    Adds one-hot encoding of the currently active model to allow
    the policy to condition on which model is currently active.
    """
    active_models = state.info['active_model']  # Shape: (batch_size,)
    
    # Create one-hot encoding of currently active model
    model_onehot = jax.nn.one_hot(active_models, self.num_models, dtype=jp.float32)  # Shape: (batch_size, num_models)
    
    # Add to both state and privileged_state observations
    new_obs = {}
    for obs_key, obs_val in state.obs.items():
      if obs_key in ['state', 'privileged_state']:
        # Append model one-hot to observation vectors
        enhanced_obs = jp.concatenate([
            obs_val,  # Original observations
            model_onehot  # Model selection as one-hot: (batch_size, num_models)
        ], axis=-1)
        new_obs[obs_key] = enhanced_obs
      else:
        new_obs[obs_key] = obs_val
    
    return state.replace(obs=new_obs)
  
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Step environment, parsing tail for FUTURE model selection.
    
    The tail of the action indicates which model to use for the NEXT reset,
    but the current step uses the model that was selected during the last reset.
    This ensures that model selection is frozen during a rollout and only
    changes at episode boundaries.
    """
    # Split action into base action and tail
    # base_action = action[..., :self.ctrl_size]
    tail = action[..., self.ctrl_size:]
    
    # Step each environment with the CURRENT (frozen) model
    def _step_single(state: mjx_env.State, action: jp.ndarray, model_idx: int) -> mjx_env.State:
      return jax.lax.switch(model_idx, self.step_fns, state, action)
    
    new_state = jax.vmap(_step_single)(state, action, state.info['active_model'])
  
    # IMPORTANT: Keep the active model unchanged during the rollout (frozen)
    new_state.info['active_model'] = state.info['active_model']
    
    # Parse tail for next model selection (will be used in NEXT reset)
    if tail.shape[-1] == self.num_models:
      # Extract argmax from tail to determine which model to use for next episode
      next_model_selection = jp.argmax(tail, axis=-1).astype(jp.int32)
      new_state.info['next_model'] = next_model_selection
    else:
      # No tail provided, keep existing next model selection
      new_state.info['next_model'] = state.info['next_model']
    
    # Add model information to observations if requested
    if self.include_model_info_in_obs:
      new_state = self._add_model_info_to_obs(new_state)
  
    return new_state
  
  @property
  def action_size(self) -> int:
    """Return the effective action size including the selection tail."""
    return self.env.action_size + self.num_models
  
  @property
  def ctrl_size(self) -> int:
    """Return the control size."""
    return self.env.action_size
  