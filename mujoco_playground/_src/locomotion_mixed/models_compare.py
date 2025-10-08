import jax
import jax.numpy as jp
from etils import epath
import numpy as np
import yaml
import pprint

import pickle
from mujoco_playground._src import mjx_env
from mujoco import mjx, MjModel
from jax.tree_util import tree_leaves

import dataclasses

# Definizione dei percorsi e caricamento dei modelli
ROOT_PATH = mjx_env.ROOT_PATH / "locomotion_mixed" / "kawaru"
FEET_ONLY_FLAT_TERRAIN_XML = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml")
FEET_ONLY_FLAT_TERRAIN_XML_2 = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain2.xml")
FEET_ONLY_FLAT_TERRAIN_XML_3 = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain3.xml")
FEET_ONLY_FLAT_TERRAIN_XML_4 = (ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain4.xml")
TEST1 = (ROOT_PATH / "xmls" / "kawaru_mjx_feetonly.xml")
TEST = (ROOT_PATH / "xmls" / "kawaru_mjx_feetonly3.xml")

XML_PATHS = [
    FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
    # FEET_ONLY_FLAT_TERRAIN_XML_2.as_posix(),
    # FEET_ONLY_FLAT_TERRAIN_XML_3.as_posix(),
    FEET_ONLY_FLAT_TERRAIN_XML_4.as_posix(),
    # TEST1.as_posix(),
    # TEST.as_posix()
]

MJ_MODEL = [MjModel.from_xml_path(path) for path in XML_PATHS]
MJX_MODEL = [mjx.put_model(mj_model, impl="jax") for mj_model in MJ_MODEL]

data_list = MJX_MODEL
# import jax
# import jax.numpy as jnp
# from etils import epath
# import numpy as np
# import yaml
# import pprint
# import dataclasses

# from mujoco_playground._src import mjx_env
# from mujoco import mjx, MjModel
# from jax.tree_util import tree_leaves


class MjFake:
    
    def __init__(self):
        pass

    def find_diffs(self, obj1, obj2):
        """
        Compares two dataclass objects and returns a dictionary of their differences.
        This function handles arbitrary levels of nested dataclasses.
        """
        diffs = {}
        aaa = jax.tree_util.tree_map(lambda x: None, obj1)
        bbb = jax.tree_util.tree_map(lambda x: None, obj2)
        
        # We use fields(obj1) to iterate and compare every field.
        # We assume obj1 and obj2 have the same fields, as they're of the same type.
        for field in dataclasses.fields(obj1):
            attr_name = field.name
            val1 = getattr(obj1, attr_name)
            val2 = getattr(obj2, attr_name)
            
            if dataclasses.is_dataclass(val1) and dataclasses.is_dataclass(val2):
                nested_diffs = self.find_diffs(val1, val2)
                
                if nested_diffs:
                    diffs[attr_name] = nested_diffs
            
            elif isinstance(val1, (jp.ndarray, np.ndarray)) and isinstance(val2, (jp.ndarray, np.ndarray)):
                if not jp.allclose(val1, val2, equal_nan=True):
                    if jp.ndim(val1) == 0:
                        print(f"❗ Value mismatch in {attr_name}: {val1} vs {val2}")
                    else:
                        condition = val1 != val2
                        idx = jp.where(condition)
                        print(f"❗ Arrays differ in {attr_name}:")
                        print(f"Arrays differ at indices: {idx}")
                        print(f"Values in val1 at these indices: {val1[idx]}")
                        print(f"Values in val2 at these indices: {val2[idx]}")
                        diffs[attr_name] = jp.stack((val1, val2))
            
            elif val1 != val2:
                diffs[attr_name] = val2
                print(f"❗ Value mismatch in {attr_name}: {val1} vs {val2}")
                print(f"{attr_name}, {val1.shape}, {val2.shape}") if isinstance(val1, (jp.ndarray, np.ndarray)) else print(f"{attr_name}, {val1}, {val2}")
                diffs[attr_name] = jp.stack((val1, val2))
        
        return diffs

    def save_diffs(self, diffs_dict, filepath):
        """
        Saves the differences dictionary to a file using pickle.
        Pickle is used because it can handle nested Python objects, including JAX arrays.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(diffs_dict, f)
        print(f"Differences saved to {filepath}")

    def load_diffs(self, filepath):
        """
        Loads the differences dictionary from a file using pickle.
        """
        with open(filepath, 'rb') as f:
            diffs_dict = pickle.load(f)
        print(f"Differences loaded from {filepath}")
        return diffs_dict

    def reduce_diffs_to_indices(self, diffs_dict):
        """
        Recursively reduces a dictionary of JAX array differences
        to only the differing indices and values.
        """
        reduced_dict = {}
        for key, value in diffs_dict.items():
            if isinstance(value, dict):
                # Handle nested dictionaries recursively
                reduced_dict[key] = self.reduce_diffs_to_indices(value)
            elif isinstance(value, jp.ndarray) and value.shape[0] == 2:
                # Assuming the diffs were stacked as in find_diffs
                val1, val2 = value[0], value[1]
                
                # Check for differences, handling NaNs
                # isclose is better than != for floating-point values
                condition = ~jp.isclose(val1, val2, equal_nan=True)
                
                if jp.any(condition):
                    # Get the indices where values differ
                    indices = jp.where(condition)
                    
                    # Store the indices and the new values from val2
                    reduced_dict[key] = {
                        'indices': indices,
                        'values': jp.stack(val1[indices], val2[indices])
                    }
            elif isinstance(value, jp.ndarray) and value.shape[0] != 2:
                # Handle single values or arrays that were not stacked
                # This could be from a non-array value difference
                reduced_dict[key] = value

        return reduced_dict


if __name__ == "__main__":

    data_diff = MjFake()
    
    diff_file_path = f"{epath.Path(__file__).parent}/model_diffs.pkl"
    reduced_diff_file_path = f"{epath.Path(__file__).parent}/model_diffs_reduced.pkl"
    
    # Find and save differences
    diff_results = data_diff.find_diffs(data_list[0], data_list[1])
    data_diff.save_diffs(diff_results, diff_file_path)
    data_diff.save_max_shapes(diff_results)
    
    # Reduce the differences and save them
    reduced_diffs = data_diff.reduce_diffs_to_indices(diff_results)
    data_diff.save_diffs(reduced_diffs, reduced_diff_file_path)

    # Now, load the reduced differences and apply them (demonstration)
    loaded_reduced_diffs = data_diff.load_diffs(reduced_diff_file_path)

    # Example of how to apply these changes with JAX
    def apply_reduced_diffs(obj, reduced_diffs):
        updates = {}
        for key, value in reduced_diffs.items():
            if 'indices' in value and 'values' in value:
                # Get the original array to modify
                original_array = getattr(obj, key)
                
                # Create a new array with the updates
                updated_array = original_array.at[value['indices']].set(value['values'])
                
                updates[key] = updated_array
            elif isinstance(value, dict):
                updates[key] = apply_reduced_diffs(getattr(obj, key), value)
            else:
                updates[key] = value # Non-array value
        
        return obj.replace(**updates)

    print("\n--- Applying loaded reduced differences ---")
    current_model = data_list[0]
    
    # We would need to traverse the whole dataclass structure
    # to apply the updates correctly. This is a simplified example.
    
    # Let's assume for simplicity we are just modifying the top-level keys
    # of the model's _impl dataclass.
    
    # new_model = current_model.replace(_impl=apply_reduced_diffs(current_model._impl, loaded_reduced_diffs['_impl']))
    
    # The actual application needs to be more robust, potentially using tree_map.
    