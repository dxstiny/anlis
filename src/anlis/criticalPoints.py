# -*- coding: utf-8 -*-
"""functions to determine critical points and extrema"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import List
import sympy as sp


def criticalPoints(function: sp.Function,
                   x: sp.Symbol = sp.Symbol("x")) -> List[sp.Number]:
    """Returns the critical points of a function of one variable.
    :param function: function of one variable
    :param x: variable
    :return: critical points of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import criticalPoints
    >>> x = symbols('x')
    >>> criticalPoints(x**2, x)
    [0]
    """
    return sp.solve(sp.diff(function, x), x, dict=True)

def minimas(function: sp.Function,
            x: sp.Symbol = sp.Symbol("x")) -> List[sp.Number]:
    """Returns the minimas of a function of one variable.
    :param function: function of one variable
    :param x: variable
    :return: minimas of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import minimas
    >>> x = symbols('x')
    >>> minimas(x**2, x)
    [0]
    """
    return [ point
             for point in criticalPoints(function, x)
             if sp.diff(function, x, 2).subs(point) > 0 ]

def maximas(function: sp.Function,
            x: sp.Symbol = sp.Symbol("x")) -> List[sp.Number]:
    """Returns the maximas of a function of one variable.
    :param function: function of one variable
    :param x: variable
    :return: maximas of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import maximas
    >>> x = symbols('x')
    >>> maximas(-x**2, x)
    [0]
    """
    return [ point
             for point in criticalPoints(function, x)
             if sp.diff(function, x, 2).subs(point) < 0]
