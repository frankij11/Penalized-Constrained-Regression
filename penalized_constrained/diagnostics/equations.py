"""
Model equation formatting utilities.

Generates human-readable equations for linear models and extracts
source code from custom prediction functions.

Supports docstring-based equation definitions:
    Equation: y = exp(b0 + b1*x1)
    Equation-LaTeX: \\hat{y} = e^{\\beta_0 + \\beta_1 x_1}
"""

import ast
import inspect
import re
from typing import Optional, Callable, List, Tuple
import numpy as np

from .dataclasses import ModelEquation


def parse_equation_from_docstring(func: Callable) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract equation definitions from a function's docstring.

    Looks for special tags:
    - Equation: <text equation>
    - Equation-LaTeX: <latex equation>

    Parameters
    ----------
    func : Callable
        The function to extract equation from

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        (text_equation, latex_equation) - either or both may be None
    """
    docstring = inspect.getdoc(func)
    if not docstring:
        return None, None

    text_eq = None
    latex_eq = None

    # Parse Equation: tag (text equation)
    text_match = re.search(r'^Equation:\s*(.+)$', docstring, re.MULTILINE)
    if text_match:
        text_eq = text_match.group(1).strip()

    # Parse Equation-LaTeX: tag
    latex_match = re.search(r'^Equation-LaTeX:\s*(.+)$', docstring, re.MULTILINE)
    if latex_match:
        latex_eq = latex_match.group(1).strip()

    return text_eq, latex_eq


def extract_return_expressions(func: Callable) -> List[str]:
    """
    Extract return expressions from a function using AST parsing.

    For return statements that return a variable (like `return ans`),
    traces back to find the last assignment to that variable.

    Parameters
    ----------
    func : Callable
        The function to analyze

    Returns
    -------
    List[str]
        List of unique return expression strings
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Find the function definition node
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break

    if func_def is None:
        return []

    # Collect all assignments in the function (for variable tracing)
    assignments = {}
    for node in ast.walk(func_def):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Store the assignment expression for this variable
                    try:
                        assignments[target.id] = ast.unparse(node.value)
                    except Exception:
                        assignments[target.id] = None

    # Collect return expressions
    expressions = []
    common_result_vars = {'ans', 'result', 'res', 'out', 'output', 'y', 'y_pred', 'pred', 'prediction'}

    for node in ast.walk(func_def):
        if isinstance(node, ast.Return) and node.value is not None:
            try:
                # Check if return value is a simple variable
                if isinstance(node.value, ast.Name):
                    var_name = node.value.id
                    # If it's a common result variable, trace back to assignment
                    if var_name.lower() in common_result_vars or var_name in assignments:
                        if var_name in assignments and assignments[var_name]:
                            expr = f"{var_name} = {assignments[var_name]}"
                            if expr not in expressions:
                                expressions.append(expr)
                            continue

                # Otherwise, just unparse the return expression
                expr = ast.unparse(node.value)
                if expr not in expressions:
                    expressions.append(expr)
            except Exception:
                continue

    return expressions


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

        # Try to extract equation from docstring
        docstring_text, docstring_latex = parse_equation_from_docstring(prediction_fn)

        # Extract return expressions via AST parsing
        return_expressions = extract_return_expressions(prediction_fn)

        # Build text field - always show both docstring equation and parsed expressions
        text_parts = []
        if docstring_text:
            text_parts.append(f"Equation: {docstring_text}")
        else:
            text_parts.append("Custom prediction function")

        if return_expressions:
            parsed_label = "Parsed: "
            indent = " " * len(parsed_label)
            for i, expr in enumerate(return_expressions):
                if i == 0:
                    text_parts.append(f"{parsed_label}{expr}")
                else:
                    text_parts.append(f"{indent}{expr}")

        text = "\n".join(text_parts)

        return ModelEquation(
            text=text,
            latex=docstring_latex,
            source=source,
            is_custom=True
        )

    # Standard linear model
    coef_ = getattr(model, 'coef_', None)
    intercept_ = getattr(model, 'intercept_', 0.0)
    # Use coef_names_in_ for coefficient names in equations
    coef_names = getattr(model, 'coef_names_in_', None)
    fit_intercept = getattr(model, 'fit_intercept', True)
    feature_names = getattr(model, 'feature_names_in_', coef_names)

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
