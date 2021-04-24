import numpy as np


def ws(x: list, y: list) -> float:
    """Calculates the value of WS similarity coefficient
    Parameters
    ----------
    x: list
        List of values for similarity comparison
    y: list
        List of values for similarity comparison
    Returns
    -------
        Value of WS similarity coefficient
    """
    n = len(x)
    x = np.asarray(x)
    y = np.asarray(y)
    return 1 - sum(np.float_power(2, -x) * np.abs(x - y) / np.maximum(np.abs(1 - x), np.abs(n - x)))


def rw(x: list, y: list) -> float:
    """Calculates the value of the weighted Spearman correlation coefficient
    Parameters
    ----------
    x: list
        List of values for similarity comparison
    y: list
        List of values for similarity comparison
    Returns
    -------
         Value of Spearman weighted correlation coefficient
    """
    n = len(x)
    x = np.asarray(x)
    y = np.asarray(y)
    return 1 - 6 * sum(((x - y) ** 2) * (2 * n - x - y + 2)) / (n ** 4 + n ** 3 - n ** 2 - n)
