import jax
import jax.numpy as jnp
from etils import epath
import numpy as np
import yaml
import pprint

from mujoco_playground._src import mjx_env
from mujoco import mjx, MjModel
from jax import numpy as jp

import dataclasses

# Definizione dei percorsi e caricamento dei modelli
ROOT_PATH = mjx_env.ROOT_PATH / "locomotion_mixed" / "kawaru"
FEET_ONLY_FLAT_TERRAIN_XML = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml")
FEET_ONLY_FLAT_TERRAIN_XML_2 = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain2.xml")
FEET_ONLY_FLAT_TERRAIN_XML_3 = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain3.xml")

XML_PATHS = [
    FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
    # FEET_ONLY_FLAT_TERRAIN_XML_2.as_posix(),
    FEET_ONLY_FLAT_TERRAIN_XML_3.as_posix(),
]

MJ_MODEL = [MjModel.from_xml_path(path) for path in XML_PATHS]
MJX_MODEL = [mjx.put_model(mj_model) for mj_model in MJ_MODEL]

data_list = []

for mjx_model in MJX_MODEL:
    qpos = jp.zeros(mjx_model.nq)
    qvel = jp.zeros(mjx_model.nv)
    data = mjx_env.init(mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:19])
    data_list.append(data)

class MjxDataFake:
    
    def __init__(self):
        pass

    def find_diffs(self, obj1, obj2):
        """
        Compares two dataclass objects and returns a dictionary of their differences.
        This function handles arbitrary levels of nested dataclasses.
        """
        diffs = {}
        
        # We use fields(obj1) to iterate and compare every field.
        # We assume obj1 and obj2 have the same fields, as they're of the same type.
        for field in dataclasses.fields(obj1):
            attr_name = field.name
            val1 = getattr(obj1, attr_name)
            val2 = getattr(obj2, attr_name)
            
            # We skip the 'contact' attribute, as per your previous request.
            if attr_name == "contact":
                pass

            if dataclasses.is_dataclass(val1) and dataclasses.is_dataclass(val2):
                # If it's a nested dataclass, we make a recursive call.
                # We don't need a current_dict parameter anymore.
                nested_diffs = self.find_diffs(val1, val2)
                
                # If the nested call found any differences, we add them to our dict.
                # if nested_diffs:
                diffs[attr_name] = nested_diffs
            
            elif isinstance(val1, (jnp.ndarray, np.ndarray)) and isinstance(val2, (jnp.ndarray, np.ndarray)):
                if val1.shape != val2.shape:
                    diffs[attr_name] = val2.shape
                    print(f"⚠️ Different shapes in {attr_name}: {val1.shape} vs {val2.shape}")
            
            elif val1 != val2:
                diffs[attr_name] = val2
                print(f"❗ Value mismatch in {attr_name}: {val1} vs {val2}")
            else:
                diffs[attr_name] = val2

        return diffs
    
    def save_max_shapes(self, dict_shapes):
        """
        Saves the dictionary of max shapes to a Python file.
        """
        with open(f"{epath.Path(__file__).parent}/max_shapes.py", "w") as f:
            f.write("import numpy as np \n")
            f.write("max_shapes = \\\n")
            f.write(pprint.pformat(dict_shapes, indent=1))
            print("Saved maximum shapes to max_shapes.py")

    def load_max_shapes(self, filepath):
        """
        Loads a dictionary of max shapes from a Python file.
        """
        namespace = {}
        with open(filepath, "r") as f:
            file_content = f.read()
            exec(file_content, namespace)
        
        return namespace['max_shapes']
    
    def pad_data(self, obj_to_pad: dataclasses, target_shapes: dict):
        """
        Recursively pads arrays in a dataclass based on a dictionary of target shapes.
        """
        updates = {}
        for field in dataclasses.fields(obj_to_pad):
            attr_name = field.name
            current_val = getattr(obj_to_pad, attr_name)

            # Check if the attribute name exists in the target shapes dictionary
            if attr_name in target_shapes:
                target_val = target_shapes[attr_name]
                
                if dataclasses.is_dataclass(current_val) and isinstance(target_val, dict):
                    # If it's a nested dataclass, recursively call pad_data
                    updates[attr_name] = self.pad_data(current_val, target_val)
                elif isinstance(current_val, (jnp.ndarray, np.ndarray)) and isinstance(target_val, (tuple, list)):
                    current_shape = current_val.shape
                    if current_shape != target_val:
                        padding_needed = [(0, t - s) for s, t in zip(current_shape, target_val)]
                        padded_val = jnp.pad(current_val, padding_needed, mode='constant', constant_values=0)
                        updates[attr_name] = padded_val
                    else:
                        updates[attr_name] = current_val
                else:
                    updates[attr_name] = target_val
            else:
                updates[attr_name] = current_val
        
        return obj_to_pad.replace(**updates)
    

class MjModelFake:

    def __init__(self):
        pass


if __name__ == "__main__":

    data_diff = MjxDataFake()

    max_shapes = data_diff.find_diffs(data_list[0], data_list[1])
    # Run the comparison

    print("\nComparison complete.")
    print(max_shapes)

    data_diff.save_max_shapes(max_shapes)
    
    del max_shapes

    max_shapes = data_diff.load_max_shapes(f"{epath.Path(__file__).parent}/max_shapes.py")
    # print(max_shapes)

    data_to_pad = data_list[0]
    target_shapes = max_shapes

    new_data = data_diff.pad_data(data_to_pad, target_shapes)
    
    # Now, compare the padded object with the second object
    print("Starting comparison of new_data and data_list[1]...")
    if not data_diff.find_diffs(new_data, data_list[1]):
        print("✅ No differences found after padding.")
    else:
        print("Confronto completato.")

    pass