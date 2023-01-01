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
    return sp.solve(gradient(function, *variables), *variables, dict=True)

def minimas(function: sp.Function,
            *variables: sp.Symbol) -> List[Dict[sp.Symbol, sp.Number]]:
    """Returns the minimas of a function of any number of variables.
    :param function: function of two or more variables
    :param variables: variables
    :return: minimas of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import minimas
    >>> x, y = symbols('x y')
    >>> minimas(x**2 + y**2, x, y)
    [{x: 0, y: 0}]
    """
    return [ point
             for point in criticalPoints(function, *variables)
             if determinant(function, list(variables)).subs(point) > 0 ]

def maximas(function: sp.Function,
            *variables: sp.Symbol) -> List[Dict[sp.Symbol, sp.Number]]:
    """Returns the maximas of a function of any number of variables.
    :param function: function of two or more variables
    :param variables: variables
    :return: maximas of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import maximas
    >>> x, y = symbols('x y')
    >>> maximas(x**2 + y**2, x, y)
    []
    """
    return [ point
             for point in criticalPoints(function, *variables)
             if determinant(function, list(variables)).subs(point) < 0 ]
