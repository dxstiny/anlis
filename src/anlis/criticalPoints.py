# -*- coding: utf-8 -*-
"""functions to determine critical points and extrema"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import List, Dict
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

def minima(function: sp.Function,
            x: sp.Symbol = sp.Symbol("x")) -> List[sp.Number]:
    """Returns the minima of a function of one variable.
    :param function: function of one variable
    :param x: variable
    :return: minima of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import minima
    >>> x = symbols('x')
    >>> minima(x**2, x)
    [0]
    """
    return [ point
             for point in criticalPoints(function, x)
             if sp.diff(function, x, 2).subs(point) > 0 ]

def maxima(function: sp.Function,
            x: sp.Symbol = sp.Symbol("x")) -> List[sp.Number]:
    """Returns the maxima of a function of one variable.
    :param function: function of one variable
    :param x: variable
    :return: maxima of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import maxima
    >>> x = symbols('x')
    >>> maxima(-x**2, x)
    [0]
    """
    return [ point
             for point in criticalPoints(function, x)
             if sp.diff(function, x, 2).subs(point) < 0]

def turningPoints(function: sp.Function,
                  x: sp.Symbol = sp.Symbol("x")) -> List[sp.Number]:
    """Returns the turning points of a function of one variable.
    :param function: function of one variable
    :param x: variable
    :return: turning points of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import turningPoints
    >>> x = symbols('x')
    >>> turningPoints(x**2, x)
    [0]
    """
    points: List[Dict[sp.Symbol, sp.Number]] = sp.solve(sp.diff(function, x, 2), x, dict=True)
    return [ point[x]
             for point in points
             if sp.diff(function, x, 3).subs(point) != 0 ]

def saddlePoints(function: sp.Function,
                 x: sp.Symbol = sp.Symbol("x")) -> List[sp.Number]:
    """Returns the saddle points of a function of one variable.
    :param function: function of one variable
    :param x: variable
    :return: saddle points of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.criticalPoints import saddlePoints
    >>> x = symbols('x')
    >>> saddlePoints(x**2, x)
    []
    """
    points = turningPoints(function, x)
    return [ point
             for point in points
             if sp.diff(function, x).subs(x, point) == 0 ]
