# -*- coding: utf-8 -*-
"""a toolset for dealing with arithmetic sequences & series"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import Dict
import sympy as sp

def fromElements(knownElements: Dict[int, float],
                 counter: sp.Symbol = sp.Symbol("n")) -> sp.Expr:
    """Determine the arithmetic series with the given elements.
    :param knownElements: The known elements of the series. -> {n: a_n}
    :return: The arithmetic series. -> a_n = a_1 + (n - 1) * d
    """
    assert len(knownElements) >= 2, "The number of known elements must be at least 2."

    knownIndices = list(knownElements.keys())

    # Determine the common difference.
    commonDifference = (knownElements[knownIndices[1]] - knownElements[knownIndices[0]]) / (knownIndices[1] - knownIndices[0]) # pylint: disable=line-too-long

    # Determine the first element.
    firstElement = knownElements[knownIndices[0]] - (knownIndices[0] - 1) * commonDifference

    # Determine the arithmetic series.
    return firstElement + (counter - 1) * commonDifference
