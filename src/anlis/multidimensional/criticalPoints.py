# -*- coding: utf-8 -*-
"""functions to determine critical points and extrema of multidimensional functions"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import List, Dict
import sympy as sp
from anlis.multidimensional.derivative import gradient, determinant


def criticalPoints(function: sp.Function,
                   *variables: sp.Symbol) -> List[Dict[sp.Symbol, sp.Number]]:
    """Returns the critical points of a function of any number of variables.
    :param function: function of two or more variables
    :param variables: variables
    :return: critical points of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import criticalPoints
    >>> x, y = symbols('x y')
    >>> criticalPoints(x**2 + y**2, x, y)
    [{x: 0, y: 0}]
    """
    points = [ ]
    g = gradient(function, *variables)
    points.extend(sp.solve(g, *variables, dict=True))

    if "sqrt" in str(g):
        print("Warning: sqrt in gradient. There may be more critical points.")
        print("Critical points are where the gradient is zero or undefined.")
        print("Gradient: ", g)

    denoms = sp.solvers.solvers.denoms(g, *variables)
    for denom in denoms:
        for sol in sp.solve(denom, *variables, dict=True):
            for var in variables:
                if var not in sol:
                    continue
                if not sol[var].is_real:
                    break
            else:
                points.append(sol)

    return points

def minima(function: sp.Function,
           *variables: sp.Symbol) -> List[Dict[sp.Symbol, sp.Number]]:
    """Returns the minima of a function of any number of variables.
    :param function: function of two or more variables
    :param variables: variables
    :return: minima of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import minima
    >>> x, y = symbols('x y')
    >>> minima(x**2 + y**2, x, y)
    [{x: 0, y: 0}]
    """
    return [ point
             for point in criticalPoints(function, *variables)
             if determinant(function, list(variables)).subs(point) > 0 ]

def maxima(function: sp.Function,
           *variables: sp.Symbol) -> List[Dict[sp.Symbol, sp.Number]]:
    """Returns the maxima of a function of any number of variables.
    :param function: function of two or more variables
    :param variables: variables
    :return: maxima of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import maxima
    >>> x, y = symbols('x y')
    >>> maxima(x**2 + y**2, x, y)
    []
    """
    return [ point
             for point in criticalPoints(function, *variables)
             if determinant(function, list(variables)).subs(point) < 0 ]
