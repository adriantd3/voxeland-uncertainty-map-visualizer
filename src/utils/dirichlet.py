import math
from typing import Iterable, Union

Number = Union[float, int]


def digamma(x: float) -> float:
    """
    Aproximación de la función digamma (ψ) usando la serie de Stirling.
    Para x < 1, usa recurrencia: ψ(x) = ψ(x+1) - 1/x.
    """
    # Coeficientes de la serie de Stirling
    c1 = 1.0 / 12.0
    c2 = -1.0 / 120.0
    c3 = 1.0 / 252.0
    c4 = -1.0 / 240.0
    c5 = 1.0 / 132.0
    c6 = -691.0 / 32760.0
    c7 = 1.0 / 12.0

    if x < 1.0:
        # Recursión para valores pequeños
        return digamma(x + 1.0) - 1.0 / x

    # Serie de Stirling
    inv = 1.0 / x
    inv2 = inv * inv
    inv4 = inv2 * inv2
    inv6 = inv4 * inv2
    inv8 = inv4 * inv4
    inv10 = inv8 * inv2
    inv12 = inv6 * inv6

    return (math.log(x)
            - 0.5 * inv
            - c1 * inv2
            + c2 * inv4
            - c3 * inv6
            + c4 * inv8
            - c5 * inv10
            + c6 * inv12
            + c7 * inv10)


def expected_shannon_entropy(alpha_value: Iterable[Number]) -> float:
    """
    Calcula la entropía de Shannon esperada bajo una distribución de Dirichlet
    con parámetros alpha = [α1, α2, ..., αK], ignorando aquellos αi = 0.

    E[H] = ψ(A) - (1 / A) * sum_i (α_i * ψ(α_i)), donde A = sum_i α_i.
    """
    # Filtrar ceros y convertir a float
    alphas = [float(a) for a in alpha_value if a != 0]

    if not alphas:
        raise ValueError("El vector alpha no puede contener solo ceros.")

    A = sum(alphas)
    sum_alpha_psi = sum(a * digamma(a) for a in alphas)

    return digamma(A) - (sum_alpha_psi / A)


# Ejemplo de uso
if __name__ == "__main__":
    # Parámetros de ejemplo
    alpha = [177,1,243,74,2000,8,4,1,0.5,0.5,1]  # incluye un cero que se ignora
    h = expected_shannon_entropy(alpha)
    print(f"Entropía de Shannon esperada: {h:.6f}")
