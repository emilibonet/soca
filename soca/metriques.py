import numpy as np


def variance_coefficient(data: np.ndarray) -> float:
    """
    Calculate the coefficient of variation (CV) of a list.
    """
    if len(data) == 0:
        return 0.0
    mean = np.mean(data)
    std_dev = np.std(data)
    if mean == 0:
        return 0.0
    return std_dev / mean