import json
import re
from typing import Iterable, Union, Dict
from src.utils.dirichlet import expected_shannon_entropy

def get_instances_entropy(json_path: str) -> Dict[int, float]:
    """
    Lee un archivo JSON con la estructura dada y devuelve un dict:
      {instance_id: expected_shannon_entropy(results_values)}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entropies: Dict[int, float] = {}
    instances = data.get("instances", {})

    for inst_id, inst_data in instances.items():
        # Extrae el número de "objXX"
        match = re.search(r'\d+', inst_id)
        if not match:
            continue  # o lanza una excepción si es obligatorio
        inst_num = int(match.group())
        results = inst_data.get("results", {})
        values = list(results.values())
        entropies[inst_num] = expected_shannon_entropy(values)

    return entropies

if __name__ == "__main__":
    # Ejemplo de uso
    json_path = "json_map/206_post_dis.json"
    entropies = get_instances_entropy(json_path)
    for inst_id, entropy in entropies.items():
        print(f"Instance ID: {inst_id}, Entropy: {entropy:.4f}")