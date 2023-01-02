# -*- coding: utf-8 -*-
"""a toolset for dealing with numerical integral approximation"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


import numpy as np
import sympy as sp


def riemannSum(function: sp.Function,
               a: float,
               b: float,
               n: int,
               upper: bool = False,
               x: sp.Symbol = sp.Symbol("x")) -> float:
    """
    Riemann sum for approximating the integral of f(x) in [a, b].
    :param f: function f(x)
    :param a: left endpoint of interval
    :param b: right endpoint of interval
    :param n: number of subintervals
    :param upper: if True, use upper (right) Riemann sum
    :return: approximation of integral
    """
    f = sp.lambdify(x, function, "numpy")
    h = (b-a)/n
    xx = np.linspace(a, b, n+1)
    y = f(xx)
    if upper:
        return h * sum(y[1:])
    return h * sum(y[:-1])

def trapezoidalRule(function: sp.Function,
                    a: float,
                    b: float,
                    n: int,
                    x: sp.Symbol = sp.Symbol("x")) -> float:
    """
    Trapezoidal rule for approximating the integral of f(x) in [a, b].
    :param f: function f(x)
    :param a: left endpoint of interval
    :param b: right endpoint of interval
    :param n: number of subintervals
    :return: approximation of integral
    """
    f = sp.lambdify(x, function, "numpy")
    h = (b-a)/n
    xx = np.linspace(a, b, n+1)
    y = f(xx)
    return h/2 * (y[0] + 2*sum(y[1:-1]) + y[-1])

def simpsonsRule(function: sp.Function,
                 a: float,
                 b: float,
                 n: int,
                 x: sp.Symbol = sp.Symbol("x")) -> float:
    """
    Simpson's rule for approximating the integral of f(x) in [a, b].
    :param f: function f(x)
    :param a: left endpoint of interval
    :param b: right endpoint of interval
    :param n: number of subintervals
    :return: approximation of integral
    """
    f = sp.lambdify(x, function, "numpy")
    h = (b-a)/n
    xx = np.linspace(a, b, n+1)
    y = f(xx)
    return h/3 * (y[0] + 4*sum(y[1:-1:2]) + 2*sum(y[2:-1:2]) + y[-1])
