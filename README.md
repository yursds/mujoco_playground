# MuJoCo Playground

![Banner for playground](https://github.com/google-deepmind/mujoco_playground/blob/main/assets/banner.png?raw=true)

A comprehensive suite of GPU-accelerated environments for robot learning research and sim-to-real, built with [MuJoCo MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx).

Features include:

- Classic control environments from `dm_control`.
- Quadruped and bipedal locomotion environments.
- Non-prehensile and dexterous manipulation environments.
- Vision-based support available via [Madrona-MJX](https://github.com/shacklettbp/madrona_mjx).

For more details, check out the project [website](https://playground.mujoco.org/).

> [!NOTE]
> We now support training with both the MuJoCo MJX JAX implementation, as well as the [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) implementation at HEAD. See MuJoCo 3.3.5 [release notes](https://mujoco.readthedocs.io/en/stable/changelog.html#version-3-3-5-august-8-2025) under `MJX` for more details.

## Installation

You can install MuJoCo Playground directly from PyPI:

```sh
pip install playground
```

> [!WARNING]
> The `playground` release may depend on pre-release versions of `mujoco` and
> `warp-lang`, in which case you can try `pip install playground
> --extra-index-url=https://py.mujoco.org
> --extra-index-url=https://pypi.nvidia.com/warp-lang/`.
> If there are still version mismatches, please open a github issue, and install
> from source.

### From Source
1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), a faster alternative to `pip`
2. Create and activate a virtual environment: 
    ``` 
    git clone -b test https://yursds/mujoco_playground.git && cd mujoco_playground
    uv venv --python 3.11 
    source .venv/bin/activate
    ```
3. Install dependecies: 
    ```
    uv pip install -e .[all]
    uv pip install -U "jax[cuda12]"
    ```
    * Verify GPU backend: `python -c "import jax; print(jax.default_backend())"` should print gpu
    * Verify installation (and download Menagerie): `python -c "import mujoco_playground"`

### Reproducibility / GPU Precision Issues
Users with NVIDIA Ampere architecture GPUs (e.g., RTX 30 and 40 series) may experience reproducibility [issues](https://github.com/google-deepmind/mujoco_playground/issues/86) in mujoco_playground due to JAX’s default use of TF32 for matrix multiplications. This lower precision can adversely affect RL training stability. To ensure consistent behavior with systems using full float32 precision (as on Turing GPUs), please run `export JAX_DEFAULT_MATMUL_PRECISION=highest` in your terminal before starting your experiments (or add it to the end of `~/.bashrc`).

## Running from CLI
For basic usage, navigate to the repo's directory and run:
```bash
python learning/train_jax_ppo.py --env_name CartpoleBalance
```

### Training Visualization

To interactively view trajectories throughout training with [rscope](https://github.com/Andrew-Luo1/rscope/tree/main), install it (`uv pip install rscope`) and run:

```
python learning/train_jax_ppo.py --env_name PandaPickCube --rscope_envs 16 --run_evals=False --deterministic_rscope=True
# In a separate terminal
python -m rscope
```

## FAQ

### How can I contribute?

Get started by installing the library and exploring its features! Found a bug? Report it in the issue tracker. Interested in contributing? If you are a developer with robotics experience, we would love your help—check out the [contribution guidelines](CONTRIBUTING.md) for more details.

### Reproducibility / GPU Precision Issues

Users with NVIDIA Ampere architecture GPUs (e.g., RTX 30 and 40 series) may experience reproducibility [issues](https://github.com/google-deepmind/mujoco_playground/issues/86) in mujoco_playground due to JAX’s default use of TF32 for matrix multiplications. This lower precision can adversely affect RL training stability. To ensure consistent behavior with systems using full float32 precision (as on Turing GPUs), please run `export JAX_DEFAULT_MATMUL_PRECISION=highest` in your terminal before starting your experiments (or add it to the end of `~/.bashrc`).

To reproduce results using the same exact learning script as used in the paper, run the brax training script which is available [here](https://github.com/google/brax/blob/1ed3be220c9fdc9ef17c5cf80b1fa6ddc4fb34fa/brax/training/learner.py#L1). There are slight differences in results when using the `learning/train_jax_ppo.py` script, see the issue [here](https://github.com/google-deepmind/mujoco_playground/issues/171) for more context.

## Citation

If you use Playground in your scientific works, please cite it as follows:

```bibtex
@misc{mujoco_playground_2025,
  title = {MuJoCo Playground: An open-source framework for GPU-accelerated robot learning and sim-to-real transfer.},
  author = {Zakka, Kevin and Tabanpour, Baruch and Liao, Qiayuan and Haiderbhai, Mustafa and Holt, Samuel and Luo, Jing Yuan and Allshire, Arthur and Frey, Erik and Sreenath, Koushil and Kahrs, Lueder A. and Sferrazza, Carlo and Tassa, Yuval and Abbeel, Pieter},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/google-deepmind/mujoco_playground}
}
```

## License and Disclaimer

The texture used in the rough terrain for the locomotion environments is from [Polyhaven](https://polyhaven.com/a/rock_face) and licensed under [CC0](https://creativecommons.org/public-domain/cc0/).

All other content in this repository is licensed under the Apache License, Version 2.0. A copy of this license is provided in the top-level [LICENSE](LICENSE) file in this repository. You can also obtain it from https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.
