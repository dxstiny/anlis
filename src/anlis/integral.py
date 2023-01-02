# -*- coding: utf-8 -*-
"""a toolset for dealing with numerical integral approximation"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import Union
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

def arc(function: sp.Function,
        x: sp.Symbol = sp.Symbol("x")) -> sp.Function:
    """
    Arc length function for approximating the integral of sqrt(1 + f'(x)^2) in [a, b].
    :param f: function f(x)
    :param x: variable
    :return: arc length function
    """
    df = function.diff(x)
    return sp.sqrt(1 + df**2)

def arcLength(function: sp.Function,
              a: float,
              b: float,
              x: sp.Symbol = sp.Symbol("x")) -> float:
    """
    Arc length for approximating the integral of sqrt(1 + f'(x)^2) in [a, b].
    :param f: function f(x)
    :param a: left endpoint of interval
    :param b: right endpoint of interval
    :param n: number of subintervals
    :return: approximation of integral
    """
    return sp.integrate(arc(function, x), (x, a, b))

def archimedeanSpiralLength(phi1: float,
                            phi2: float,
                            r: Union[sp.Expr, float] = sp.Symbol("phi")) -> float:
    """
    Archimedean spiral length in [phi1, phi2].
    :param phi1: left endpoint of interval
    :param phi2: right endpoint of interval
    :param r: radius
    :return: length
    """
    phi = sp.Symbol("phi")

    if isinstance(r, sp.Expr):
        dr = r.diff(phi)
    else:
        dr = 0

    g = sp.sqrt(r**2 + dr**2)
    return sp.integrate(g, (phi, phi1, phi2))
