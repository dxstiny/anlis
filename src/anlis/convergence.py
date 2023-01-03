# -*- coding: utf-8 -*-
"""a toolset for dealing with convergence & divergence"""
from __future__ import annotations
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import Optional, Union

import sympy as sp
from sympy import Rational


class Convergence:
    """A class for representing convergence and divergence of a sequence."""
    def __init__(self,
                 does: bool = False,
                 to: Optional[Union[float, Rational]] = None) -> None:
        self._does = does
        self._to = to

    @property
    def does(self) -> bool:
        """Returns True if the sequence converges, False otherwise."""
        return self._does

    @property
    def doesNot(self) -> bool:
        """Returns True if the sequence diverges, False otherwise."""
        return not self._does

    @property
    def to(self) -> Union[float, Rational]:
        """Returns the value the sequence converges to."""
        assert self.does
        return self._to

    @property
    def toRational(self) -> Rational:
        """Returns the value the sequence converges to as a Rational."""
        return Rational(self._to)

    @property
    def toFloat(self) -> float:
        """Returns the value the sequence converges to as a float."""
        return float(self.to)

    def __str__(self) -> str:
        if self.does:
            return f'converges to {self.to} (~{self.toFloat}))'
        return 'diverges'

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return self.does

    @staticmethod
    def convergence(to: float) -> Convergence:
        """Returns a Convergence object representing convergence to a value."""
        return Convergence(True, to)

    @staticmethod
    def divergence() -> Convergence:
        """Returns a Convergence object representing divergence."""
        return Convergence(False, None)


def regulaFalsi(f: sp.Function,
                a: float,
                b: float,
                eps: float,
                maxiter: int,
                symbol: sp.Symbol = sp.Symbol("x")) -> Optional[Convergence]:
    """
    Regula Falsi method for finding a root of f(x) = 0 in the interval [a, b].
    :param f: function f(x) = 0
    :param a: left endpoint of interval
    :param b: right endpoint of interval
    :param eps: tolerance
    :param maxiter: maximum number of iterations
    :param symbol: symbol of the function (defaults to x)
    :return: (flag, iter)

    Examples
    ========
    >>> import sympy as sp
    >>> from anlis.convergence import regulaFalsi
    >>> x = sp.Symbol("x")
    >>> regulaFalsi(x ** 2 - 5, 0.5, 2, 1e-6, 1000)
    (regula falsi) converged after 6 iterations
    Converges to 2.2360678206352302 (~2.2360678206352302)
    """
    g = sp.lambdify(symbol, f, "numpy")
    i = 1
    flag = True
    while flag:
        x = a - g(a)*(b-a) / (g(b) - g(a))
        # print ('Schnittpunkt bei 9615.12e•
        if g(a)*g(x) < 0:
            b = x # Vorzeichenwechsel in [a, x J
        else:
            a = x # Vorzeichenwechsel in [x, b]
            # print( ' Vorzeichenwechsel rechts' )
            i += 1
            flag = abs(g(x)) > eps and i < maxiter
    if i < maxiter:
        print(f"(regula falsi) converged after {i} iterations")
        return Convergence.convergence(x)

    print("(regula falsi) didn't converge after {maxiter} iterations")
    return None

def newton(f: sp.Function,
           x0: float,
           eps: float,
           maxiter: int,
           symbol: sp.Symbol = sp.Symbol("x")) -> Optional[Convergence]:
    """
    Newton method for finding a root of f(x) = 0.
    :param f: function f(x) = 0
    :param x0: initial guess
    :param eps: tolerance
    :param maxiter: maximum number of iterations
    :param symbol: symbol of the function (defaults to x)
    :return: (flag, iter)

    Examples
    ========
    >>> import sympy as sp
    >>> from anlis.convergence import newton
    >>> x = sp.Symbol("x")
    >>> newton(x ** 2 - 5, 2, 1e-6, 1000)
    (newton) converged after 4 iterations
    converges to 51841/23184 (~2.236067977915804))
    """
    i = 1
    flag = True
    while flag:
        x = x0 - f.subs(symbol, x0) / f.diff(symbol).subs(symbol, x0)
        i += 1
        flag = abs(f.subs(symbol, x)) > eps and i < maxiter
        x0 = x
    if i < maxiter:
        print(f"(newton) converged after {i} iterations")
        return Convergence.convergence(x)

    print(f"(newton) didn't converge after {maxiter} iterations")
    return None

def ratioTest(ak: sp.Function, symbol: sp.Symbol = sp.Symbol("k")) -> Optional[Convergence]:
    """
    Ratio test for convergence of a series.
    :param ak: a_k
    :param symbol: symbol of the series
    :return: convergence

    Examples
    ========

    >>> import sympy as sp
    >>> from anlis.convergence import ratioTest
    >>> k = sp.Symbol("k")
    >>> ratioTest(k / 5 ** k, k)
    Converges to 1/5
    """
    ak1 = ak.subs(symbol, symbol + 1)
    print(ak1)
    to = sp.limit(sp.Abs(ak1 / ak), symbol, sp.oo)
    print(to)

    if to == 1:
        return None
    if to > 1:
        return Convergence.divergence()
    return Convergence.convergence(to)

def rootTest(ak: sp.Function, symbol: sp.Symbol = sp.Symbol("k")) -> Optional[Convergence]:
    """
    Root test for convergence of a series.
    :param ak: a_k
    :param symbol: symbol of the series
    :return: convergence

    Examples
    ========
    >>> import sympy as sp
    >>> from anlis.convergence import rootTest
    >>> k = sp.Symbol("k")
    >>> rootTest(k / 5 ** k, k)
    Converges to 1/5
    """
    f = sp.root(sp.Abs(ak), symbol)
    to = sp.limit(f, symbol, sp.oo)

    if to == 1:
        return None
    if to > 1:
        return Convergence.divergence()
    return Convergence.convergence(to)

def radiusOfConvergence(ak: sp.Function, symbol: sp.Symbol = sp.Symbol("k")) -> Optional[float]:
    """
    Radius of convergence test for a series.
    :param ak: a_k
    :param symbol: symbol of the series
    :return: convergence

    Examples
    ========
    >>> import sympy as sp
    >>> from anlis.convergence import radiusOfConvergence
    >>> k = sp.Symbol("k")
    >>> radiusOfConvergence(k / 5 ** k, k)
    Converges to 1/5
    """
    ak1 = ak.subs(symbol, symbol + 1)
    to = sp.limit(sp.Abs(ak1 / ak), symbol, sp.oo)
    R = 1 / to
    return R
