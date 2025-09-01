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

import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple, Dict, Union

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx, MjModel
import numpy as np
from ml_collections import config_dict

from mujoco_playground._src import mjx_env

class Wrapper(mjx_env.MjxEnv):
  """Wraps an environment to allow modular transformations."""

  def __init__(self, env: Any):  # pylint: disable=super-init-not-called
    self.env = env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    return self.env.reset(rng)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return self.env.step(state, action)

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    return self.env.observation_size

  @property
  def action_size(self) -> int:
    return self.env.action_size

  @property
  def unwrapped(self) -> Any:
    return self.env.unwrapped

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self.env.mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self.env.mjx_model

  @property
  def xml_path(self) -> str:
    return self.env.xml_path

  def render(
      self,
      trajectory: List[mjx_env.State],
      height: int = 240,
      width: int = 320,
      camera: Optional[str] = None,
      scene_option: Optional[mujoco.MjvOption] = None,
      modify_scene_fns: Optional[
          Sequence[Callable[[mujoco.MjvScene], None]]
      ] = None,
  ) -> Sequence[np.ndarray]:
    return self.env.render(
        trajectory, height, width, camera, scene_option, modify_scene_fns
    )


def wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    mixed_env: bool = False,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]] = None,
    mixed_cfg: Optional[config_dict.ConfigDict] = None,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    vision: whether the environment will be vision based
    num_vision_envs: number of environments the renderer should generate,
      should equal the number of batched envs
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  if vision:
    env = MadronaWrapper(env, num_vision_envs, randomization_fn)
  elif mixed_env:
    env = MixedEnvWrapper(env, mixed_cfg, randomization_fn)
  elif randomization_fn is None:
    env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
  else:
    env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper(env)
  return env


def wrap_for_mixed_brax_training(
    env:  mjx_env.MjxEnv,
    model_cfg: config_dict.ConfigDict,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
  """Custom wrapper pattern.

  Args:
    env: environment to be wrapped
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    mjx_models: sequence of Mujoco JAX models to use
    mj_models: sequence of Mujoco Python models to use
    randomization_fn: randomization function that produces a vectorized system
      and in_axes to vmap over

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  
  from mujoco_playground._src.locomotion_mixed.kawaru.joystick import default_config
  
  env = MixedEnvWrapper(
    dummy_env=env,
    xml_paths=model_cfg.xml_paths,
    config=default_config()
    )

  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper(env)
  
  return env


class MixedEnvWrapperOLD(Wrapper):
  """Wrapper for environments that manage multiple Mujoco models."""

  def __init__(
      self,
      dummy_env: mjx_env.MjxEnv,
      xml_paths: List[str],
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      randomization_fn: Optional[Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]] = None,
  ):
    super().__init__(dummy_env)
    
    self.env_class = dummy_env.__class__
    
    self.mj_models = [MjModel.from_xml_path(path) for path in xml_paths]
    self.mjx_models = [mjx.put_model(mj_model) for mj_model in self.mj_models]
    self.config = config
    self.config_overrides = config_overrides
    
    self.num_models = len(self.mjx_models)
    self.model_indices = None
    
    self.reset_fns = [functools.partial(self._reset_for_model, i) for i in range(self.num_models)]
    self.step_fns = [functools.partial(self._step_for_model, i) for i in range(self.num_models)]
  
  def _reset_for_model(self, model_idx: int, rng_key: jax.Array) -> mjx_env.State:
    env = self.env_class(mjx_model=self.mjx_models[model_idx], mj_model=self.mj_models[model_idx], config=self.config, config_overrides=self.config_overrides)
    return env.reset(rng_key, self.mjx_models[model_idx])

  def _step_for_model(self, model_idx: int, state: mjx_env.State, action: jp.ndarray) -> mjx_env.State:
    env = self.env_class(mjx_model=self.mjx_models[model_idx], mj_model=self.mj_models[model_idx], config=self.config, config_overrides=self.config_overrides)
    return env.step(state, action, self.mjx_models[model_idx])

  def _reset_single(self, rng_key: jax.Array, model_idx: int) -> mjx_env.State:
    return jax.lax.switch(model_idx, self.reset_fns, rng_key)

  def _step_single(self, state: mjx_env.State, action: jp.ndarray, model_idx: int) -> mjx_env.State:
    return jax.lax.switch(model_idx, self.step_fns, state, action)

  @functools.partial(jax.jit, static_argnums=(0,))
  def reset(self, rng_key: jax.Array) -> mjx_env.State:
    """Resets the environment with multiple models."""
    
    if self.model_indices is None:
      # TODO the division here assumes that rng_key.shape[0] is divisible by self.num_models
      self.model_indices = jp.concatenate([jp.full(rng_key.shape[0] // self.num_models, i) for i in range(self.num_models)])
    
    initial_states = jax.vmap(self._reset_single)(rng_key, self.model_indices)
    initial_states.info['model_index'] = self.model_indices
    return initial_states

  @functools.partial(jax.jit, static_argnums=(0,))
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Steps the environment with multiple models."""
    
    new_state = jax.vmap(self._step_single)(state, action, state.info['model_index'])
    new_state.info['model_index'] = state.info['model_index']
    return new_state


class MixedEnvWrapperDiverse(Wrapper):
  """Wrapper for environments that manage multiple Mujoco models."""

  def __init__(
    self,
    dummy_env: mjx_env.MjxEnv,
    xml_paths: List[str],
    config: config_dict.ConfigDict,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    randomization_fn: Optional[Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]] = None,
  ):
    super().__init__(dummy_env)
    
    self.num_models = len(xml_paths)
    self.environments = []
    
    for path in xml_paths:
      mj_model = MjModel.from_xml_path(path)
      env = dummy_env.__class__(mj_model=mj_model, config=config, config_overrides=config_overrides)
      
      # if randomization_fn:
      #   randomized_env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
      #   self.environments.append(randomized_env)
      # else:
      #   self.environments.append(env)
      
      self.environments.append(env)
    
    self.reset_fns = [env.reset for env in self.environments]
    self.step_fns = [env.step for env in self.environments]
    self.model_indices = None
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def reset(self, rng_key: jax.Array) -> mjx_env.State:
    '''Reset the environment.'''
    
    def _reset_single(rng_key: jax.Array, model_idx: int) -> mjx_env.State:
      '''Reset a single environment.'''
      return jax.lax.switch(model_idx, self.reset_fns, rng_key)
    
    if self.model_indices is None:
      self.model_indices = jp.concatenate([jp.full(rng_key.shape[0] // self.num_models, i) for i in range(self.num_models)])
    
    initial_states = jax.vmap(_reset_single)(rng_key, self.model_indices)
    initial_states.info['model_index'] = self.model_indices
    return initial_states
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    '''Step the environment.'''

    def _step_single(state: mjx_env.State, action: jp.ndarray, model_idx: int) -> mjx_env.State:
      '''Step a single environment.'''
      return jax.lax.switch(model_idx, self.step_fns, state, action)
    
    new_state = jax.vmap(_step_single)(state, action, state.info['model_index'])
    new_state.info['model_index'] = state.info['model_index']
    return new_state


class MixedEnvWrapper(Wrapper):
  """Wrapper per ambienti che gestiscono più modelli Mujoco."""

  def __init__(
    self,
    dummy_env: mjx_env.MjxEnv,
    xml_paths: List[str],
    config: config_dict.ConfigDict,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    randomization_fn: Optional[Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]] = None,
  ):
    super().__init__(dummy_env)
    
    self.num_models = len(xml_paths)
    self.environments = []
    for path in xml_paths:
      mj_model = MjModel.from_xml_path(path)
      env = dummy_env.__class__(mj_model=mj_model, config=config, config_overrides=config_overrides)
      
      # if randomization_fn:
      #   randomized_env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
      #   self.environments.append(randomized_env)
      # else:
      #   self.environments.append(env)
      self.environments.append(env)

    self.reset_fns = tuple(env.reset for env in self.environments)
    self.step_fns = tuple(env.step for env in self.environments)

  # @functools.partial(jax.jit, static_argnums=(0,))
  def reset(self, rng_key: jax.Array) -> mjx_env.State:
    '''Reset the environment.'''
    
    def _reset_single(rng_key: jax.Array, model_idx: int) -> mjx_env.State:
      '''Reset a single environment.'''
      return jax.lax.switch(model_idx, self.reset_fns, rng_key)
    
    dim_size = rng_key.shape[0]

    # Genera gli indici dei modelli per il batch
    indices = jp.arange(self.num_models)
    replicated_indices = jp.repeat(indices, dim_size // self.num_models, total_repeat_length=dim_size)
    
    initial_states = jax.vmap(_reset_single)(rng_key, replicated_indices)
    initial_states.info['model_index'] = replicated_indices
    return initial_states
  
  # @functools.partial(jax.jit, static_argnums=(0,))
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    '''Step the environment.'''

    def _step_single(state: mjx_env.State, action: jp.ndarray, model_idx: int) -> mjx_env.State:
      '''Step a single environment.'''
      return jax.lax.switch(model_idx, self.step_fns, state, action)
    
    new_state = jax.vmap(_step_single)(state, action, state.info['model_index'])
    new_state.info['model_index'] = state.info['model_index']
    return new_state


# import copy
# class MixedEnvWrapperCOPY(Wrapper):
#   """Wrapper per ambienti che gestiscono più modelli Mujoco."""

#   def __init__(
#     self,
#     dummy_env: mjx_env.MjxEnv,
#     xml_paths: List[str],
#     config: config_dict.ConfigDict,
#     config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
#     randomization_fn: Optional[Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]] = None,
#   ):
#     super().__init__(dummy_env)
    
#     self.num_models = len(xml_paths)
#     self.mjx_list = []
#     self.mj_list = []

#     for path in xml_paths:
#       mj_model = MjModel.from_xml_path(path)
#       env = dummy_env.__class__(mj_model=mj_model, config=config, config_overrides=config_overrides)
#       self.mjx_list.append(env.mjx_model)
#       self.mj_list.append(env.mj_model)

#     self.reset_fns = tuple(self._env_fn(mjx, mj).reset for mjx, mj in zip(self.mjx_list, self.mj_list))
#     self.step_fns = tuple(self._env_fn(mjx, mj).step for mjx, mj in zip(self.mjx_list, self.mj_list))

#   def _env_fn(self, mjx_model: mjx.Model, mj_model: mujoco.MjModel) -> mjx_env.MjxEnv:
#     env = copy.deepcopy(self.env)
#     env.unwrapped._mjx_model = mjx_model
#     env.unwrapped._mj_model = mj_model
#     return env
  
#   # @functools.partial(jax.jit, static_argnums=(0,))
#   def reset(self, rng_key: jax.Array) -> mjx_env.State:
#     '''Reset the environment.'''
    
#     def _reset_single(rng_key: jax.Array, model_idx: int) -> mjx_env.State:
#       '''Reset a single environment.'''
#       return jax.lax.switch(model_idx, self.reset_fns, rng_key)
    
#     dim_size = rng_key.shape[0]

#     # Genera gli indici dei modelli per il batch
#     indices = jp.arange(self.num_models)
#     replicated_indices = jp.repeat(indices, dim_size // self.num_models, total_repeat_length=dim_size)
    
#     initial_states = jax.vmap(_reset_single)(rng_key, replicated_indices)
#     initial_states.info['model_index'] = replicated_indices
#     return initial_states
  
#   # @functools.partial(jax.jit, static_argnums=(0,))
#   def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
#     '''Step the environment.'''

#     def _step_single(state: mjx_env.State, action: jp.ndarray, model_idx: int) -> mjx_env.State:
#       '''Step a single environment.'''
#       return jax.lax.switch(model_idx, self.step_fns, state, action)
    
#     new_state = jax.vmap(_step_single)(state, action, state.info['model_index'])
#     new_state.info['model_index'] = state.info['model_index']
#     return new_state


# class MixedEnvWrapper(Wrapper):
#   """Wrapper per ambienti che gestiscono più modelli Mujoco."""

#   def __init__(
#     self,
#     dummy_env: mjx_env.MjxEnv,
#     xml_paths: List[str],
#     config: config_dict.ConfigDict,
#     config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
#     randomization_fn: Optional[Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]] = None,
#   ):
#     super().__init__(dummy_env)
    
#     self.num_models = len(xml_paths)
#     mjx_model_list = []

#     # Carichiamo i modelli Mujoco e li convertiamo in JAX-compatibili
#     for path in xml_paths:
#       mj_model = MjModel.from_xml_path(path)
#       env = dummy_env.__class__(mj_model=mj_model, config=config, config_overrides=config_overrides)
#       mjx_model_list.append(env.mjx_model)

#     # Memorizziamo solo la tupla dei modelli JAX. Questo è un pytree.
#     self.mjx_model_list = tuple(mjx_model_list)

#   @functools.partial(jax.jit, static_argnums=(0,))
#   def reset(self, rng_key: jax.Array) -> mjx_env.State:
    
#     # Funzione pura che sarà vettorizzata. Prende il modello come argomento.
#     def reset_fn(rng, model):
#       return self.env.unwrapped.reset(rng, model=model)

#     dim_size = rng_key.shape[0]

#     # Genera gli indici dei modelli per il batch
#     indices = jp.arange(self.num_models)
#     replicated_indices = jp.repeat(indices, dim_size // self.num_models, total_repeat_length=dim_size)

#     # Seleziona i modelli JAX compatibili usando un'operazione JAX (`jp.take`)
#     models_to_reset = jp.take(jp.array(self.mjx_model_list), replicated_indices, axis=0)

#     # `vmap` ora opera su RNG e sui modelli, senza toccare `self.env`
#     state = jax.vmap(reset_fn, in_axes=[0, 0])(rng_key, models_to_reset)
    
#     state.info['model_index'] = replicated_indices
#     return state
  
#   @functools.partial(jax.jit, static_argnums=(0,))
#   def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

#     # Funzione pura che sarà vettorizzata. Prende il modello come argomento.
#     def step_fn(s, a, model):
#       return self.env.unwrapped.step(s, a, model=model)
    
#     # Seleziona i modelli JAX compatibili in base all'indice nello stato
#     models_to_step = jp.take(jp.array(self.mjx_model_list), state.info['model_index'], axis=0)

#     # `vmap` ora opera su stato, azione e modelli
#     res = jax.vmap(step_fn, in_axes=[0, 0, 0])(state, action, models_to_step)
    
#     res.info['model_index'] = state.info['model_index']
#     return res


class MixedEnvWrapperpppp(Wrapper):
  """Brax wrapper for random model selection from a list."""

  def __init__(
      self,
      dummy_env: mjx_env.MjxEnv,
      xml_paths: List[str],
      config: config_dict.ConfigDict,
      # randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):
    super().__init__(dummy_env)
    mjx_model_list = []
    
    for path in xml_paths:
      mj_model = MjModel.from_xml_path(path)
      env = dummy_env.__class__(mj_model=mj_model, config=config)
      mjx_model_list.append(env.mjx_model)
    
    # 1. Convert the list of models into a single, batched PyTree.
    # This is the key to making this work with vmap.
    self._mjx_model_v = jax.tree.map(lambda *a: jp.stack(a), *mjx_model_list)
    
    # 2. Create the in_axes.
    # Since we have a batch of models, all axes must be mapped.
    self._in_axes = jax.tree.map(lambda x: 0, self._mjx_model_v)
    
    # 3. Store the number of models for later use.
    self.num_models = len(mjx_model_list)

  def _env_fn(self, mjx_model: mjx_env.MjxEnv) -> mjx_env.MjxEnv:
    # A helper function to create an environment from a specific model.
    env = self.env
    env.unwrapped._mjx_model = mjx_model
    return env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    
    # A JIT-compiled function to reset a single environment.
    def reset_single(mjx_model, rng):
      env = self._env_fn(mjx_model=mjx_model)
      return env.reset(rng)

    # 1. Randomly select model indices for the batch.
    rng, key = jax.random.split(rng)
    batch_size = rng.shape[0]
    model_choices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.num_models)
    
    # 2. Use jax.vmap to select the correct models for each environment in the batch.
    # jax.lax.index_take is the efficient JAX equivalent of list indexing.
    selected_models_v = jax.tree.map(lambda x: jax.lax.index_take(x, model_choices, axes=(0,)), self._mjx_model_v)

    # 3. Vmap the reset operation over the selected models.
    state = jax.vmap(reset_single, in_axes=[self._in_axes, 0])(selected_models_v, rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # This step is the same as your original, but now it will work correctly
    # because the input state and model are properly batched.
    
    def step_single(mjx_model, s, a):
      env = self._env_fn(mjx_model=mjx_model)
      return env.step(s, a)

    # Since the reset operation already created a batched model,
    # we need to re-select the models to pass to vmap.
    model_indices = state.info['model_index'] # This assumes model_index is stored in state.info
    
    selected_models_v = jax.tree.map(lambda x: jax.lax.index_take(x, model_indices, axes=(0,)), self._mjx_model_v)

    res = jax.vmap(step_single, in_axes=[self._in_axes, 0, 0])(selected_models_v, state, action)
    return res


class BraxAutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = self.env.reset(rng)
    state.info['first_state'] = state.data
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)

    data = jax.tree.map(where_done, state.info['first_state'], state.data)
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    return state.replace(data=data, obs=obs)


class BraxDomainRandomizationVmapWrapper(Wrapper):
  """Brax wrapper for domain randomization."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):
    super().__init__(env)
    self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

  def _env_fn(self, mjx_model: mjx.Model) -> mjx_env.MjxEnv:
    env = self.env
    env.unwrapped._mjx_model = mjx_model
    return env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    def reset(mjx_model, rng):
      env = self._env_fn(mjx_model=mjx_model)
      return env.reset(rng)

    state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    def step(mjx_model, s, a):
      env = self._env_fn(mjx_model=mjx_model)
      return env.step(s, a)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
        self._mjx_model_v, state, action
    )
    return res


def _identity_vision_randomization_fn(
    mjx_model: mjx.Model, num_worlds: int
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary fields for the Madrona memory buffer copy."""
  in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_type': 0,
      'light_castshadow': 0,
      'light_cutoff': 0,
  })
  mjx_model = mjx_model.tree_replace({
      'geom_rgba': jp.repeat(
          jp.expand_dims(mjx_model.geom_rgba, 0), num_worlds, axis=0
      ),
      'geom_matid': jp.repeat(
          jp.expand_dims(jp.repeat(-1, mjx_model.geom_matid.shape[0], 0), 0),
          num_worlds,
          axis=0,
      ),
      'geom_size': jp.repeat(
          jp.expand_dims(mjx_model.geom_size, 0), num_worlds, axis=0
      ),
      'light_pos': jp.repeat(
          jp.expand_dims(mjx_model.light_pos, 0), num_worlds, axis=0
      ),
      'light_dir': jp.repeat(
          jp.expand_dims(mjx_model.light_dir, 0), num_worlds, axis=0
      ),
      'light_type': jp.repeat(
          jp.expand_dims(mjx_model.light_type, 0), num_worlds, axis=0
      ),
      'light_castshadow': jp.repeat(
          jp.expand_dims(mjx_model.light_castshadow, 0), num_worlds, axis=0
      ),
      'light_cutoff': jp.repeat(
          jp.expand_dims(mjx_model.light_cutoff, 0), num_worlds, axis=0
      ),
  })
  return mjx_model, in_axes


def _supplement_vision_randomization_fn(
    mjx_model: mjx.Model,
    randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    num_worlds: int,
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary missing fields for the Madrona memory buffer copy."""
  mjx_model, in_axes = randomization_fn(mjx_model)

  required_fields = [
      'geom_rgba',
      'geom_matid',
      'geom_size',
      'light_pos',
      'light_dir',
      'light_type',
      'light_castshadow',
      'light_cutoff',
  ]

  for field in required_fields:
    if getattr(in_axes, field) is None:
      in_axes = in_axes.tree_replace({field: 0})
      val = -1 if field == 'geom_matid' else getattr(mjx_model, field)
      mjx_model = mjx_model.tree_replace({
          field: jp.repeat(jp.expand_dims(val, 0), num_worlds, axis=0),
      })
  return mjx_model, in_axes


class MadronaWrapper:
  """Wraps a MuJoCo Playground to be used in Brax with Madrona."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      num_worlds: int,
      randomization_fn: Optional[
          Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
      ] = None,
  ):
    if not randomization_fn:
      randomization_fn = functools.partial(
          _identity_vision_randomization_fn, num_worlds=num_worlds
      )
    else:
      randomization_fn = functools.partial(
          _supplement_vision_randomization_fn,
          randomization_fn=randomization_fn,
          num_worlds=num_worlds,
      )
    self._env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    self.num_worlds = num_worlds

    # For user-made DR functions, ensure that the output model includes the
    # needed in_axes and has the correct shape for madrona initialization.
    required_fields = [
        'geom_rgba',
        'geom_matid',
        'geom_size',
        'light_pos',
        'light_dir',
        'light_type',
        'light_castshadow',
        'light_cutoff',
    ]
    for field in required_fields:
      assert hasattr(self._env._in_axes, field), f'{field} not in in_axes'
      assert (
          getattr(self._env._mjx_model_v, field).shape[0] == num_worlds
      ), f'{field} shape does not match num_worlds'

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""
    return self._env.reset(rng)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Run one timestep of the environment's dynamics."""
    return self._env.step(state, action)

  def __getattr__(self, name):
    """Delegate attribute access to the wrapped instance."""
    return getattr(self._env.unwrapped, name)
