# -*- coding: utf-8 -*-
"""a toolset for dealing with taylor values & series"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import Union, Tuple

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def TnValue(f: sp.Function, # pylint: disable=invalid-name
            x0: float,
            n: int,
            symbol: sp.Symbol = sp.Symbol("x")) -> sp.Function:
    """
    Returns the nth Taylor value of f(x) at x0.
    :param f: function f(x)
    :param x0: point x0
    :param n: nth Taylor value
    :param symbol: symbol of x
    :return: nth Taylor value of f(x) at x0
    """
    df = sp.diff(f, symbol, n).subs(symbol, x0)
    return df / sp.factorial(n) * (symbol - x0)**n

def Tn(f: sp.Function, # pylint: disable=invalid-name
       x0: float,
       n: int,
       symbol: sp.Symbol = sp.Symbol("x")) -> sp.Function:
    """
    Returns the nth Taylor polynomial of f(x) at x0.
    :param f: function f(x)
    :param x0: point x0
    :param n: nth Taylor polynomial
    :param symbol: symbol of x
    :return: nth Taylor polynomial of f(x) at x0
    """
    tn = 0
    for i in range(n + 1):
        tn += TnValue(f, x0, i, symbol)
    return tn

def TnErrorFunction(f: sp.Function, # pylint: disable=invalid-name
                    x0: float,
                    n: int,
                    symbol: sp.Symbol = sp.Symbol("x")) -> sp.Function:
    """
    Returns the nth Taylor error function of f(x) at x0.
    :param f: function f(x)
    :param x0: point x0
    :param n: nth Taylor error
    :param symbol: symbol of x
    :return: nth Taylor error function of f(x) at x0
    """
    return f - Tn(f, x0, n, symbol)

def TnError(f: sp.Function, # pylint: disable=invalid-name
            x0: float,
            a: float,
            b: float,
            n: int,
            symbol: sp.Symbol = sp.Symbol("x")) -> float:
    """
    Returns the nth Taylor error of f(x) at x0.
    :param f: function f(x)
    :param x0: point x0
    :param a: left bound
    :param b: right bound
    :param n: nth Taylor error
    :param symbol: symbol of x
    :return: nth Taylor error of f(x) at x0
    """
    errf = TnErrorFunction(f, x0, n, symbol)
    xx = np.linspace(a, b, 100)
    errfnp = sp.lambdify(symbol, errf, "numpy")
    yy = errfnp(xx)
    return np.max(np.abs(yy))

def lagrangeRemainder(f: sp.Function,
                      x0: float,
                      a: float,
                      b: float,
                      n: int,
                      symbol: sp.Symbol = sp.Symbol("x")) -> sp.Function:
    """
    Returns the Lagrange remainder of f(x) at x0.
    :param f: function f(x)
    :param x0: point x0
    :param a: left bound
    :param b: right bound
    :param n: nth Lagrange remainder
    :param symbol: symbol of x
    :return: nth Lagrange remainder of f(x) at x0
    """
    n += 1

    x = sp.Symbol("x")
    M = sp.maximum((x - x0)**n, x, sp.Interval(abs(a), abs(b)))
    m = sp.root(M, n) + x0

    dfn = sp.diff(f, symbol, n)

    if dfn.is_negative:
        dfn = -dfn

    g = dfn * M / sp.factorial(n)

    T = sp.maximum(g, symbol, sp.Interval(x0, m))

    if float(T) == 0:
        if n > 2:
            print(f"Warning: Lagrange Remainder is likely incorrect (T = 0) - returning for n={n - 2} instead") # pylint: disable=line-too-long
            return lagrangeRemainder(f, x0, a, b, n - 2, symbol)

        print("Warning: Lagrange Remainder is likely incorrect (T = 0) - returning 0")
        return 0
    return sp.Abs(T)

def TnMin(f: sp.Function, # pylint: disable=invalid-name
          x0: float,
          a: float,
          b: float,
          eps: float,
          symbol: sp.Symbol = sp.Symbol("x")) -> int:
    """
    Returns the minimum n such that the nth Taylor error of f(x) at x0 is less than eps.
    :param f: function f(x)
    :param x0: point x0
    :param a: left bound
    :param b: right bound
    :param eps: maximum error
    :param symbol: symbol of x
    :return: minimum n such that the nth Taylor error of f(x) at x0 is less than eps
    """
    n = 0
    while TnError(f, x0, a, b, n, symbol) > eps:
        n += 1
    return n

def taylorPlot(f: sp.Function,
               x0: float,
               n: int,
               interval: Union[sp.Interval, Tuple[float, float]] = (-1, 1),
               x: sp.Symbol = sp.Symbol("x")) -> None:
    """
    Plots the nth Taylor polynomial *and* the function f(x) on the same graph.
    blue: f(x)
    orange: T_n(x)
    :param f: function f(x)
    :param x0: point x0
    :param n: nth Taylor polynomial
    :param interval: interval to plot on
    :param x: symbol of x
    :return: None
    """
    tn = Tn(f, x0, n, x)

    if isinstance(interval, sp.Interval):
        a, b = interval.args
    else:
        a, b = interval

    xx = np.linspace(x0 + a, x0 + b, 1000)
    fnp = sp.lambdify(x, f, "numpy")
    tnp = sp.lambdify(x, tn, "numpy")
    plt.plot(xx, fnp(xx), label="f(x)")
    plt.plot(xx, tnp(xx), label=f"T_{n}(x)")
