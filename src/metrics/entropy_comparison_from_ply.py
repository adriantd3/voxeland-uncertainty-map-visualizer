import numpy as np
import sys

def compare_maps_entropy(
    original_map: str,
    comparison_map: str,
    property_name: str = "uncertainty_categories"
) -> float:
    """
    Suma la entropía total de ambos mapas y devuelve la diferencia porcentual.
    """
    def sum_property(file_path, property_name):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        header_ended = False
        properties = []
        start_index = 0

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
            raise ValueError("El archivo PLY no tiene un header válido.")

        if property_name not in properties:
            raise ValueError(f"No se encontró la propiedad '{property_name}'.")

        property_index = properties.index(property_name)
        total = 0.0

        for line in lines[start_index:start_index + num_vertices]:
            values = line.split()
            if "nan" in values[property_index]:
                continue
            total += float(values[property_index])

        return total

    total_original = sum_property(original_map, property_name)
    total_comparison = sum_property(comparison_map, property_name)

    if total_original == 0:
        raise ValueError("La suma de la entropía del mapa original es cero.")

    porcentaje = 100.0 * (total_original - total_comparison) / total_original
    return porcentaje

def get_distinct_instance_id(file_path: str) -> set:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header_ended = False
    properties = []
    start_index = 0

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
        raise ValueError("El archivo PLY no tiene un header válido.")

    if "instanceid" not in properties:
        raise ValueError("No se encontró la propiedad 'instanceid'.")

    property_index = properties.index("instanceid")
    instance_ids = set()

    for line in lines[start_index:start_index + num_vertices]:
        values = line.split()
        if "nan" in values[property_index]:
            continue
        instance_ids.add(int(values[property_index]))

    return instance_ids

if __name__ == "__main__":
    original_map = "ply_maps/map_test.ply"
    comparison_map = "ply_maps/map_test_updated.ply"
    diff = compare_maps_entropy(original_map, comparison_map, property_name="uncertainty_categories")
    instances = get_distinct_instance_id(original_map)
    print(f"Diferencia porcentual de entropía: {diff:.2f}%")
    print("IDs de instancia distintos:", instances)