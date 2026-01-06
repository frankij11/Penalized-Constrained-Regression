"""
Loss functions for penalized-constrained regression optimization.

This module provides standard loss functions and a factory function
for retrieving them by name or accepting custom callables.
"""

import numpy as np


def sspe_loss(y_true, y_pred):
    """Sum of Squared Percentage Errors (MUPE-consistent).

    SSPE = Σ((y_true - y_pred) / y_true)²

    This loss function is consistent with Mean Unsigned Percentage Error (MUPE),
    making it appropriate for cost estimation where relative errors matter
    more than absolute errors.

    Parameters
    ----------
    y_true : ndarray
        True target values.
    y_pred : ndarray
        Predicted values.

    Returns
    -------
    float
        Sum of squared percentage errors.

    Notes
    -----
    A small epsilon (1e-10) is used to avoid division by zero when
    y_true values are very close to zero.
    """
    denom = np.where(np.abs(y_true) < 1e-10, 1e-10, y_true)
    return np.sum(((y_true - y_pred) / denom) ** 2)


def sse_loss(y_true, y_pred):
    """Sum of Squared Errors.

    SSE = Σ(y_true - y_pred)²

    Standard least squares loss function.

    Parameters
    ----------
    y_true : ndarray
        True target values.
    y_pred : ndarray
        Predicted values.

    Returns
    -------
    float
        Sum of squared errors.
    """
    return np.sum((y_true - y_pred) ** 2)


def mse_loss(y_true, y_pred):
    """Mean Squared Error.

    MSE = (1/n) Σ(y_true - y_pred)²

    Normalized version of SSE.

    Parameters
    ----------
    y_true : ndarray
        True target values.
    y_pred : ndarray
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)


# Registry of built-in loss functions
LOSS_FUNCTIONS = {
    'sspe': sspe_loss,
    'sse': sse_loss,
    'mse': mse_loss,
}


def get_loss_function(loss):
    """Get a loss function by name or return a custom callable.

    Parameters
    ----------
    loss : str or callable
        Either a string name ('sspe', 'sse', 'mse') or a custom
        callable with signature loss(y_true, y_pred) -> float.

    Returns
    -------
    callable
        The loss function.

    Raises
    ------
    ValueError
        If loss is a string but not a recognized name.

    Examples
    --------
    >>> loss_fn = get_loss_function('sspe')
    >>> loss_fn([1, 2, 3], [1.1, 2.0, 2.9])  # doctest: +SKIP

    >>> custom_loss = lambda y, yhat: np.sum(np.abs(y - yhat))
    >>> loss_fn = get_loss_function(custom_loss)
    """
    if callable(loss):
        return loss

    if loss not in LOSS_FUNCTIONS:
        valid_names = list(LOSS_FUNCTIONS.keys())
        raise ValueError(
            f"Unknown loss '{loss}'. Use {valid_names} or a callable."
        )

    return LOSS_FUNCTIONS[loss]
