import pickle
import os

def get_values_within_margin(r_input, *dicts):
    result = []
    for d in dicts:
        key = closest_key(r_input, d)
        if key is not None:
            result.append(d[key])
    return tuple(result)


def closest_key(r_input, values_dict):
    closest_key = None
    closest_distance = float('inf')
    
    for key, value in values_dict.items():
        distance = abs(key - r_input)
        if distance < closest_distance:
            closest_distance = distance
            closest_key = key
    if closest_distance > 0.001:
        return 
    else:       
        return closest_key