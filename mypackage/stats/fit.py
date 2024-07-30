import numpy as np


def least_square(A: np.ndarray, y:np.ndarray) -> np.ndarray:
    """least square algorithm to fit a linear model

    Parameters
    ----------
    A : np.ndarray
        The design matrix
    y : np.ndarray
        The target vector

    Returns
    -------
    np.ndarray
        return the fitted coefficients
    """
    beta = np.linalg.inv(A.T @ A) @ A.T @ y
    return beta


def weighted_least_square(A: np.ndarray, y:np.ndarray, W: np.ndarray) -> np.ndarray:
    """weighted least square algorithm to fit a linear model

    Parameters
    ----------
    A : np.ndarray
        The design matrix
    y : np.ndarray
        The target vector
    W : np.ndarray
        The weight matrix

    Returns
    -------
    np.ndarray
        return the fitted coefficients
    """
    beta = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ y
    return beta

    