from brax.training.agents.ppo import checkpoint
from etils import epath
import jax
from mujoco_playground import registry
import mujoco
import mediapy as media
import json
from ml_collections import config_dict
from mujoco_playground._src.gait import draw_joystick_command
import functools
import numpy as np
from jax import numpy as jp


env_name = "KawaruJoystickMixed" # Go1JoystickFlatTerrain KawaruJoystickMixed
current_path = epath.Path(__file__).parent
policy_dir = epath.Path("logs/KawaruJoystickMixed-1env_min_morphology/checkpoints").as_posix() 


path_checkpoints = current_path.parent / policy_dir

# Load the policy from the specified checkpoint
checkpoint_name = "000200540160" # 000200540160 000206438400
policy_fn = checkpoint.load_policy(
    epath.Path(path_checkpoints) / checkpoint_name, 
    deterministic=True
)

# Load the config.json content as a string
env_cfg_text = (epath.Path(policy_dir) / "config.json").read_text()
env_cfg_dict = json.loads(env_cfg_text)
env_cfg = config_dict.ConfigDict(env_cfg_dict)

inference_fn = policy_fn # Usiamo la versione non-JIT per la compilazione interna
eval_env = registry.load(env_name, config=env_cfg)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(inference_fn)

# --- Funzione di Rollout JAX CORRETTA (do_rollout) ---

# Rileviamo la lunghezza dell'episodio dal config
EPISODE_LENGTH = 1000
SEED = 1

def do_rollout(rng, state):
    # Definisce la struttura dati della traiettoria per la registrazione
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )
    # 🌟 CORREZIONE 1: Includiamo anche il campo 'info' nello stato vuoto
    empty_info = {k: None for k in state.info.keys()}
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})
    empty_traj = empty_traj.replace(data=empty_data, info=empty_info) 

    def step(carry, _):
        state, rng = carry
        rng, act_key = jax.random.split(rng)
        act = jit_inference_fn(state.obs, act_key)[0]
        state = eval_env.step(state, act)
        
        # 🌟 FIX: We must include state.data.xpos and state.data.xmat 
        #         (which is implicitly part of data fields saved by tree_replace)
        traj_data = empty_traj.tree_replace({
            # NOTE: To simplify, let's just save ALL data fields required for rendering
            "data.qpos": state.data.qpos,
            "data.qvel": state.data.qvel,
            "data.time": state.data.time,
            "data.ctrl": state.data.ctrl,
            "data.mocap_pos": state.data.mocap_pos,
            "data.mocap_quat": state.data.mocap_quat,
            "data.xfrc_applied": state.data.xfrc_applied,
            "data": state.data, # <-- This saves all data fields, including xpos and xmat
            "info": {"command": state.info["command"]}, 
        })
        return (state, rng), traj_data

    # Esegue lax.scan per l'intera lunghezza dell'episodio
    (_, _), traj = jax.lax.scan(
        step, (state, rng), None, length=EPISODE_LENGTH
    )
    return traj

# --- Esecuzione e Disimpilamento dei Dati ---

rng = jax.random.split(jax.random.PRNGKey(SEED), 1)
# Reset dell'ambiente (il comando iniziale casuale è qui)
reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)

# Esecuzione JIT del rollout vettorizzato
traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)

# Disimpilamento dei dati JAX in una lista Python di stati
trajectories = [None] * 1
for i in range(1):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(EPISODE_LENGTH)
    ]

# Usiamo la prima (e unica) traiettoria
rollout = trajectories[0]

# --- Logica di Rendering ---

render_every = 2
fps = 1.0 / eval_env.dt / render_every
print(f"FPS for rendering: {fps}")

# 🌟 NUOVO LOOP per preparare i modificatori di scena dal rollout salvato
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

traj = rollout[::render_every]
mod_fns = modify_scene_fns[::render_every]
assert len(traj) == len(mod_fns)

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

frames = eval_env.render(
    traj,
    camera="track",
    modify_scene_fns=mod_fns,
    scene_option=scene_option,
    width=640,
    height=480,
)
media.write_video("test.mp4", frames, fps=fps)
print("Rollout video saved as 'test.mp4'.")