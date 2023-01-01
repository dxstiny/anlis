# -*- coding: utf-8 -*-
"""a toolset for dealing with geometric sequences & series"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import Dict
import sympy as sp

def fromElements(knownElements: Dict[int, float],
                 counter: sp.Symbol = sp.Symbol("n")) -> sp.Expr:
    """
    Determine the geometric series with the given elements.
    :param knownElements: The known elements of the series. -> {n: a_n}
    :return: The geometric series. -> a_n = a_1 * q ** (n - 1)
    """
    assert len(knownElements) >= 2, "The number of known elements must be at least 2."

    knownIndices = list(knownElements.keys())

    # Determine the common ratio.
    commonRatio = knownElements[knownIndices[1]] / knownElements[knownIndices[0]]
    neg = commonRatio < 0
    commonRatio = abs(commonRatio)
    commonRatio = sp.root(commonRatio, knownIndices[1] - knownIndices[0])
    if neg:
        commonRatio = -commonRatio

    # Determine the first element.
    firstElement = knownElements[knownIndices[0]] / commonRatio ** (knownIndices[0] - 1)

    # Determine the geometric series.
    return firstElement * commonRatio ** (counter - 1)

def sumFromElements(*elements: float) -> sp.Expr:
    """
    Determine the sum of the given elements.
    :param elements: The elements to sum.
    :return: The sum of the given elements.
    """
    series = fromElements({i + 1: elements[i] for i in range(len(elements))})
    return sp.Sum(series, (sp.Symbol("n"), 1, sp.oo)).doit()

def sumFromFirstElement(firstElement: float, q: float) -> sp.Expr:
    """
    Determine the sum of the first n elements of a geometric series.
    :param firstElement: The first element of the series.
    :param commonRatio: The common ratio of the series.
    :return: The sum of all elements of the series.
    """
    return (firstElement / (1 - q)).simplify()
