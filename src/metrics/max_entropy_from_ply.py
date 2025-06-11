import numpy as np
import sys

def get_max_property_value(file_path, property_name):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header_ended = False
    properties = []
    start_index = 0

    # Parse header
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
        elif line.startswith('property'):
            properties.append(line.split()[-1])
        elif line.startswith('end_header'):
            header_ended = True
            start_index = i + 1
            break

    if not header_ended:
        raise ValueError("The PLY file does not have a proper header ending with 'end_header'.")

    # Check if the property exists
    if property_name not in properties:
        raise ValueError(f"Property '{property_name}' not found in the point cloud.")

    property_index = properties.index(property_name)

    max_value = float('-inf')

    # Parse vertex data
    data = []
    for line in lines[start_index:start_index + num_vertices]:
        values = line.split()
        if "nan" in values[property_index]:
            continue
        property_value = float(values[property_index])
        data.append(property_value)
        if property_value > max_value:
            max_value = property_value

    quantile_value = np.quantile(np.array(data), 0.99)
    return max_value, quantile_value

def get_min_nonzero_property_value(file_path, property_name):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header_ended = False
    properties = []
    start_index = 0

    # Parse header
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
        elif line.startswith('property'):
            properties.append(line.split()[-1])
        elif line.startswith('end_header'):
            header_ended = True
            start_index = i + 1
            break

    if not header_ended:
        raise ValueError("The PLY file does not have a proper header ending with 'end_header'.")

    # Check if the property exists
    if property_name not in properties:
        raise ValueError(f"Property '{property_name}' not found in the point cloud.")

    property_index = properties.index(property_name)

    min_value = float('inf')

    # Parse vertex data
    for line in lines[start_index:start_index + num_vertices]:
        values = line.split()
        if "nan" in values[property_index]:
            continue
        property_value = float(values[property_index])
        if property_value > 0 and property_value < min_value:
            min_value = property_value

    return min_value

def get_avg_nonzero_property_value(file_path, property_name):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header_ended = False
    properties = []
    start_index = 0

    # Parse header
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
        elif line.startswith('property'):
            properties.append(line.split()[-1])
        elif line.startswith('end_header'):
            header_ended = True
            start_index = i + 1
            break

    if not header_ended:
        raise ValueError("The PLY file does not have a proper header ending with 'end_header'.")

    # Check if the property exists
    if property_name not in properties:
        raise ValueError(f"Property '{property_name}' not found in the point cloud.")

    property_index = properties.index(property_name)

    total_value = 0.0
    count = 0

    # Parse vertex data
    for line in lines[start_index:start_index + num_vertices]:
        values = line.split()
        if "nan" in values[property_index]:
            continue
        property_value = float(values[property_index])
        if property_value > 0:
            total_value += property_value
            count += 1

    return total_value / count if count > 0 else 0.0

if __name__ == "__main__":

    file_path = "maps/map_test_updated.ply"
    max_value_instances, quantile_instances = get_max_property_value(file_path, "uncertainty_instances")
    max_value_categories, quantile_categories = get_max_property_value(file_path, "uncertainty_categories")
    min_value_categories = get_min_nonzero_property_value(file_path, "uncertainty_categories")
    avg_value_categories = get_avg_nonzero_property_value(file_path, "uncertainty_categories")
    print(f"Uncertainty_instances: MAX: {max_value_instances} / 99-quantile: {quantile_instances}")
    print(f"Uncertainty_categories: MAX: {max_value_categories} / 99-quantile: {quantile_categories}")
    print(f"Uncertainty_categories: MIN: {min_value_categories}")
    print(f"Uncertainty_categories: AVG: {avg_value_categories}")
