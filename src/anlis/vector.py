# -*- coding: utf-8 -*-
"""a toolset for dealing with vectors"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import Tuple
import sympy as sp


def magnitude(*vector: sp.Function) -> sp.Function:
    """Returns the magnitude of a vector.
    :param vector: vector
    :return: magnitude of vector

    Example:
    >>> from sympy import symbols
    >>> from anlis.vectors.vectors import magnitude
    >>> x, y = symbols('x y')
    >>> magnitude(x, y)
    sqrt(x**2 + y**2)
    """
    return sp.sqrt(sum(v**2 for v in vector))

def unitVector(*vector: sp.Function) -> Tuple[sp.Function, ...]:
    """Returns the unit vector of a vector.
    :param vector: vector
    :return: unit vector of vector

    Example:
    >>> from sympy import symbols
    >>> from anlis.vectors.vectors import unitVector
    >>> x, y = symbols('x y')
    >>> unitVector(x, y)
    (x/sqrt(x**2 + y**2), y/sqrt(x**2 + y**2))
    """
    return tuple(v/magnitude(*vector) for v in vector)

def dotProduct(*vectors: Tuple[sp.Function, ...]) -> sp.Function:
    """Returns the dot product of two vectors.
    :param vectors: vectors
    :return: dot product of vectors

    Example:
    >>> from sympy import symbols
    >>> from anlis.vectors.vectors import dotProduct
    >>> x, y = symbols('x y')
    >>> dotProduct((x, y), (x, y))
    x**2 + y**2
    """
    return sum(v1*v2 for v1, v2 in zip(*vectors))
