# -*- coding: utf-8 -*-
"""a toolset for dealing with differentials and differential approximation"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import Tuple

import sympy as sp


def absoluteDifferential(function: sp.Function,
                         x0: Tuple[sp.Symbol, float],
                         dx: float) -> sp.Function:
    """
    Returns the absolute differential of a function
    :param function: function
    :param x0: point
    :param dx: differential
    :return: differential approximation of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.differential import differentialApproximation
    >>> d = symbols('d')
    >>> V = 1/12 * sp.pi * d ** 3
    >>> differentialApproximation(V, (d, 26), 0.5)
    84.5 * π
    """
    x = x0[0]
    xx = x0[1]
    df = sp.diff(function, x)
    return df.subs(x, xx) * dx


def relativeDifferential(function: sp.Function,
                         x0: Tuple[sp.Symbol, float],
                         dx: float) -> sp.Function:
    """
    Returns the relative differential of a function
    :param function: function
    :param x0: point
    :param dx: differential
    :return: relative differential approximation of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.differential import relativeDifferential
    >>> d = symbols('d')
    >>> V = 1/12 * sp.pi * d ** 3
    >>> relativeDifferential(V, (d, 26), 0.01)
    0.0576923076923077
    """
    return absoluteDifferential(function, x0, dx) / function.subs(x0[0], x0[1])
