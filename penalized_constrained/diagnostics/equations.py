"""
Model equation formatting utilities.

Generates human-readable equations for linear models and extracts
source code from custom prediction functions.
"""

import inspect
from typing import Optional, Callable, List, Union
import numpy as np

from .dataclasses import ModelEquation


def get_callable_source(func: Callable) -> Optional[str]:
    """
    Extract source code from a callable, handling lambdas gracefully.

    Parameters
    ----------
    func : Callable
        The function to extract source from

    Returns
    -------
    Optional[str]
        Source code if available, signature if not, or placeholder for lambdas
    """
    if func is None:
        return None

    try:
        # Try to get full source code
        source = inspect.getsource(func)
        return source.strip()
    except (OSError, TypeError):
        # Source not available (lambda, built-in, etc.)
        pass

    # Try to get at least the signature
    try:
        sig = inspect.signature(func)
        func_name = getattr(func, '__name__', '<callable>')

        # Check if it's a lambda
        if func_name == '<lambda>':
            return f"<lambda>{sig}"

        return f"{func_name}{sig}"
    except (ValueError, TypeError):
        pass

    # Last resort
    func_name = getattr(func, '__name__', '<callable>')
    return f"<{func_name}>"


def format_linear_equation(
    coef_: np.ndarray,
    intercept_: float,
    feature_names: Optional[List[str]],
    fit_intercept: bool,
    decimals: int = 4
) -> dict:
    """
    Format standard linear model equation.

    Parameters
    ----------
    coef_ : np.ndarray
        Coefficient values
    intercept_ : float
        Intercept value
    feature_names : Optional[List[str]]
        Names for features, or None for x1, x2, ...
    fit_intercept : bool
        Whether intercept was fitted
    decimals : int
        Number of decimal places

    Returns
    -------
    dict
        {'text': str, 'latex': str} with formatted equations
    """
    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(len(coef_))]
    else:
        feature_names = list(feature_names)

    # Build text equation
    terms = []
    latex_terms = []

    if fit_intercept:
        terms.append(f"{intercept_:.{decimals}f}")
        latex_terms.append(f"{intercept_:.{decimals}f}")

    for name, coef in zip(feature_names, coef_):
        if np.abs(coef) < 1e-10:
            continue  # Skip zero coefficients

        # Determine sign
        if coef >= 0:
            sign = " + " if terms else ""
            latex_sign = " + " if latex_terms else ""
        else:
            sign = " - " if terms else "-"
            latex_sign = " - " if latex_terms else "-"

        abs_coef = abs(coef)
        terms.append(f"{sign}{abs_coef:.{decimals}f}*{name}")

        # Sanitize name for LaTeX (replace underscores)
        latex_name = name.replace("_", r"\_")
        latex_terms.append(f"{latex_sign}{abs_coef:.{decimals}f} \\cdot {latex_name}")

    text_eq = "y = " + "".join(terms) if terms else "y = 0"
    latex_eq = r"\hat{y} = " + "".join(latex_terms) if latex_terms else r"\hat{y} = 0"

    return {'text': text_eq, 'latex': latex_eq}


def format_model_equation(model) -> ModelEquation:
    """
    Generate equation representation for any model type.

    Parameters
    ----------
    model : PenalizedConstrainedRegression or PenalizedConstrainedCV
        Fitted model

    Returns
    -------
    ModelEquation
        Equation representation with text, latex, and optionally source code
    """
    # Check for custom prediction function
    prediction_fn = getattr(model, 'prediction_fn', None)

    if prediction_fn is not None:
        # Custom prediction function - extract source
        source = get_callable_source(prediction_fn)

        # Also try to get the loss function source if custom
        loss_fn = getattr(model, 'loss', None)
        loss_source = None
        if callable(loss_fn):
            loss_source = get_callable_source(loss_fn)
            if loss_source:
                source = f"# Prediction function:\n{source}\n\n# Loss function:\n{loss_source}"

        return ModelEquation(
            text="Custom prediction model (see source code)",
            latex=None,
            source=source,
            is_custom=True
        )

    # Standard linear model
    coef_ = getattr(model, 'coef_', None)
    intercept_ = getattr(model, 'intercept_', 0.0)
    feature_names = getattr(model, 'feature_names_in_', None)
    fit_intercept = getattr(model, 'fit_intercept', True)

    if coef_ is None:
        return ModelEquation(
            text="Model not fitted",
            latex=None,
            source=None,
            is_custom=False
        )

    eq_dict = format_linear_equation(
        coef_=coef_,
        intercept_=intercept_,
        feature_names=feature_names,
        fit_intercept=fit_intercept
    )

    return ModelEquation(
        text=eq_dict['text'],
        latex=eq_dict['latex'],
        source=None,
        is_custom=False
    )


def format_loss_function(model) -> str:
    """
    Format the loss function description.

    Parameters
    ----------
    model : fitted model
        The model to describe

    Returns
    -------
    str
        Description of the loss function
    """
    loss = getattr(model, 'loss', 'unknown')

    if isinstance(loss, str):
        loss_descriptions = {
            'sspe': 'Sum of Squared Percentage Errors (SSPE)',
            'sse': 'Sum of Squared Errors (SSE)',
            'mse': 'Mean Squared Error (MSE)',
        }
        return loss_descriptions.get(loss, loss)
    elif callable(loss):
        source = get_callable_source(loss)
        return f"Custom: {source}"
    else:
        return str(loss)
