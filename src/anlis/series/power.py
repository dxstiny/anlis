# -*- coding: utf-8 -*-
"""a toolset for dealing with geometric sequences & series"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


import sympy as sp

def determine(series: sp.Expr,
              x: sp.Symbol = sp.Symbol("x")) -> sp.Expr:
    """
    Determine a power series.
    :param function: function to determine, must be a power series, with x and k
    :param x: symbol of the function
    :return: series
    """
    series1 = series.subs(x, 1)
    return series1
