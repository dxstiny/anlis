# -*- coding: utf-8 -*-
"""a toolset for dealing with polynomials"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import Optional, Tuple, Union, Set
import sympy as sp


def poles(function: sp.Function, x: sp.Symbol = sp.Symbol("x")) -> Set[sp.Number]:
    """
    Returns the poles of a function.
    :param function: the function
    :return: the poles
    """
    d = sp.solvers.solvers.denoms(function, x)
    ps = set()
    for i in d:
        ps.update(sp.solve(i, x))
    return ps

def zeros(function: sp.Function, x: sp.Symbol = sp.Symbol("x")) -> Set[sp.Number]:
    """
    Returns the zeros of a function.
    :param function: the function
    :return: the zeros
    """
    n, _ = sp.fraction(function)
    return set(sp.Poly(n, x).all_roots())

def isContinuous(function: sp.Function,
                 interval: Optional[Union[sp.Interval,
                                          Tuple[float, float]]] = None) -> bool:
    """
    Checks if a function is continuous.
    :param function: the function
    :param interval: the interval
    """
    if interval is None:
        interval = sp.Interval(-sp.oo, sp.oo)

    if isinstance(interval, tuple):
        interval = sp.Interval(*interval)

    ps = poles(function)

    for pole in ps:
        if pole in interval:
            return False

    return True
