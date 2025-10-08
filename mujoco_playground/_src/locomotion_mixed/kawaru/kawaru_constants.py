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
"""Defines Unitree Kawaru quadruped constants."""

from etils import epath
from mujoco_playground._src import mjx_env
from mujoco import mjx, MjModel

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion_mixed" / "kawaru"
FEET_ONLY_FLAT_TERRAIN_MIN_XML = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain_min.xml")

FEET_ONLY_FLAT_TERRAIN_ACT_FAKE_XML = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain_act_fake.xml")
FEET_ONLY_FLAT_TERRAIN_XML = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml")
FEET_ONLY_FLAT_TERRAIN_XML_2 = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain2.xml")
FEET_ONLY_FLAT_TERRAIN_XML_3 = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain3.xml")
FEET_ONLY_FLAT_TERRAIN_XML_4 = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain4.xml")
FEET_ONLY_FLAT_TERRAIN_XML_5 = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain5.xml")

FEET_ONLY_ROUGH_TERRAIN_XML = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml")
FULL_FLAT_TERRAIN_XML = (ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml")
FULL_COLLISIONS_FLAT_TERRAIN_XML = (ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml")

XML_PATHS = [
    # FEET_ONLY_FLAT_TERRAIN_ACT_FAKE_XML.as_posix(),
    FEET_ONLY_FLAT_TERRAIN_MIN_XML.as_posix(),
    # FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
    # FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
    # FEET_ONLY_FLAT_TERRAIN_XML_2.as_posix(),
    # FEET_ONLY_FLAT_TERRAIN_XML_3.as_posix(),
    # FEET_ONLY_FLAT_TERRAIN_XML_4.as_posix(),
    # FEET_ONLY_FLAT_TERRAIN_XML_5.as_posix(),
]

MJ_MODEL = [MjModel.from_xml_path(path) for path in XML_PATHS]
# MJX_MODEL = [mjx.put_model(mj_model, impl='warp') for mj_model in MJ_MODEL]

# def task_to_xml(task_name: str) -> epath.Path:
#   return {
#       "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
#       "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
#   }[task_name]

FEET_SITES = ["FR", "FL", "RR", "RL"]

FEET_GEOMS = ["FR", "FL", "RR", "RL"]

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "trunk"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"


# def load_max_shapes(filepath):
#     """
#     Loads a dictionary of max shapes from a Python file.
#     """
#     namespace = {}
#     with open(filepath, "r") as f:
#         file_content = f.read()
#         exec(file_content, namespace)
    
#     return namespace['max_shapes']
# MJX_DATA_MAX_SHAPES = load_max_shapes(ROOT_PATH / "max_shapes.py")

