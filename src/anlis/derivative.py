# -*- coding: utf-8 -*-
"""a toolset for dealing with derivatives"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import Tuple, Optional
import sympy as sp


def newtonRaphson(function: sp.Function,
                  x0: float,
                  n: int = 10,
                  x: sp.Symbol = sp.Symbol("x")) -> float:
    """
    Newton-Raphson method for finding a root of f(x) = 0.
    :param f: function f(x)
    :param x0: initial guess
    :param n: number of iterations
    :param x: variable
    :return: sequence of approximations
    """
    f = sp.lambdify(x, function, "numpy")
    df = sp.lambdify(x, function.diff(x), "numpy")
    x = x0
    for _ in range(n):
        x = x - f(x) / df(x)
    return x

def newtonRaphsonWithEpsilon(function: sp.Function,
                             x0: float,
                             epsilon: float,
                             maxiter: int = 1000,
                             x: sp.Symbol = sp.Symbol("x")) -> Optional[Tuple[int, float]]:
    """
    Newton-Raphson method for finding a root of f(x) = 0.
    :param f: function f(x)
    :param x0: initial guess
    :epsilon: the error tolerance
    :param x: variable
    :return: number of iterations and the root (n, x)
    """
    f = sp.lambdify(x, function, "numpy")
    df = sp.lambdify(x, function.diff(x), "numpy")
    x = x0
    for i in range(maxiter):
        x = x - f(x) / df(x)
        if abs(f(x)) < epsilon:
            return i, x
    return None

def oneSidedDerivative(function: sp.Function,
                       x0: float,
                       direction: str = "+",
                       x: sp.Symbol = sp.Symbol("x")) -> sp.Number:
    """
    One-sided derivative
    :param f: function f(x)
    :param x0: point at which to approximate derivative
    :param direction: direction of approximation ("+" or "-")
    :param x: variable
    :return: derivative
    """
    dx = sp.Symbol("dx")
    nom = function.subs(x, x0 + dx) - function.subs(x, x0)
    denom = dx
    return sp.limit(nom / denom, dx, 0, direction)

def existsDerivative(function: sp.Function,
                     x0: float,
                     x: sp.Symbol = sp.Symbol("x")) -> bool:
    """
    Check if a derivative exists at a point
    :param f: function f(x)
    :param x0: point at which to check
    :param x: variable
    :return: True if derivative exists, False otherwise
    """
    return oneSidedDerivative(function, x0, "+", x) == oneSidedDerivative(function, x0, "-", x)

def curvature(function: sp.Function,
              x: sp.Symbol = sp.Symbol("x")) -> sp.Expr:
    """
    Curvature at a point
    :param f: function f(x)
    :param x: variable
    :return: curvature
    """
    return function.diff(x, 2) / (1 + function.diff(x) ** 2) ** (3 / 2)

def radiusOfCurvature(function: sp.Function,
                      x: sp.Symbol = sp.Symbol("x")) -> sp.Expr:
    """
    Radius of curvature at a point
    :param f: function f(x)
    :param x: variable
    :return: radius of curvature
    """
    return 1 / curvature(function, x)

def centreOfCurvature(function: sp.Function,
                      x: sp.Symbol = sp.Symbol("x")) -> Tuple[sp.Expr, sp.Expr]:
    """
    Centre of curvature at a point
    :param f: function f(x)
    :param x: variable
    :return: centre of curvature
    """
    df = function.diff(x)
    tmpl = (1 + df ** 2) / function.diff(x, 2)
    mx = x - df * tmpl
    my = function + tmpl
    return mx, my
