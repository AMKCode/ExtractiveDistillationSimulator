from scipy.spatial.distance import euclidean
import pickle

def approximate_value(dictionary, search_value, metric=euclidean):
    """
    Approximates the value to search for keys in the final_dict based on a given metric (default is Euclidean).
    
    Parameters:
    - final_dict (dict): The final consolidated dictionary.
    - search_value (float): The value to search for.
    - metric (function): The distance metric to use for approximation (default is Euclidean distance).
    
    Returns:
    - closest_values (list): List of values corresponding to the closest key.
    """
    # Find the key that is closest to the search_value
    closest_key = min(dictionary.keys(), key=lambda k: metric([k], [search_value]))
    
    # Get the values corresponding to the closest key
    closest_values = dictionary.get(closest_key, None)
    
    return closest_values

def consolidate_dicts_with_source(dict_list, tolerance, save_path=None):
    """
    Consolidates keys that are close enough from multiple dictionaries into a single dictionary.
    Also keeps track of which original dictionary each value came from.
    
    Parameters:
    - dict_list (list): List of dictionaries to be consolidated.
    - tolerance (float): Tolerance level for considering keys as close enough.
    - save_path (str, optional): Path to save the consolidated dictionary as a pickle file.

    Returns:
    - final_dict (dict): Consolidated dictionary with source indices.
    
    Structure of final_dict:
    - Key (float): The consolidated key.
    - Value (dict): A dictionary containing two lists:
        - 'values': A list of values corresponding to the key from the original dictionaries.
        - 'source_indices': A list of integers indicating which original dictionary each value came from.
    """
    # Initialize the final dictionary
    final_dict = {}
    
    # Iterate through each dictionary and each key
    for idx, d in enumerate(dict_list):
        for key, value in d.items():
            # Check if this key is close to an existing key in the final dictionary
            close_key = None
            for existing_key in final_dict.keys():
                if abs(existing_key - key) <= tolerance:
                    close_key = existing_key
                    break
            
            # If a close key is found, check if this dictionary already contributed a value
            if close_key is not None:
                if idx not in final_dict[close_key]['source_indices']:
                    final_dict[close_key]['values'].append(value)
                    final_dict[close_key]['source_indices'].append(idx)
            # Otherwise, create a new key-value pair
            else:
                final_dict[key] = {'values': [value], 'source_indices': [idx]}
                
    # Optionally save the final dictionary as a pickle file
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(final_dict, f)
            
    return final_dict

