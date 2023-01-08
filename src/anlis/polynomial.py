# -*- coding: utf-8 -*-
"""a toolset for dealing with polynomials"""
from __future__ import annotations
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import Optional, Tuple, Union, Set
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
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

def realZeros(function: sp.Function, x: sp.Symbol = sp.Symbol("x")) -> Set[sp.Number]:
    """
    Returns the real zeros of a function.
    :param function: the function
    :return: the real zeros
    """
    return set(filter(lambda z: z.is_real, zeros(function, x)))

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

def isEven(function: sp.Function, x: sp.Symbol = sp.Symbol("x")) -> bool:
    """
    Checks if a function is even.
    :param function: the function
    :return: whether the function is even
    """
    return function.subs(x, -x) == function


class ParabolaBuilder:
    """A builder for parabolas."""
    def __init__(self,
                 a: sp.Number = 0,
                 b: sp.Number = 0,
                 c: sp.Number = 0) -> None:
        self._a = a
        self._b = b
        self._c = c

    @classmethod
    def fromABC(cls, a: sp.Number, b: sp.Number, c: sp.Number) -> ParabolaBuilder:
        """
        Creates a parabola builder from a, b and c.
        :param a: the a
        :param b: the b
        :param c: the c
        """
        return cls(a, b, c)

    @classmethod
    def fromVertex(cls, a: sp.Number, u: sp.Number, v: sp.Number) -> ParabolaBuilder:
        """
        Creates a parabola builder from the vertex.
        :param u: the x coordinate of the vertex
        :param v: the y coordinate of the vertex
        """
        return cls(a, -2*a*u, a*u**2 + v)

    @classmethod
    def fromVertexAndPoint(cls,
                           u: sp.Number,
                           v: sp.Number,
                           x: sp.Number,
                           y: sp.Number) -> ParabolaBuilder:
        """
        Creates a parabola builder from the vertex and a point.
        :param u: the x coordinate of the vertex
        :param v: the y coordinate of the vertex
        :param x: the x coordinate of the point
        :param y: the y coordinate of the point
        """
        a = (y - v) / (x - u)**2
        return cls(a, -2*a*u, a*u**2 + v)

    @classmethod
    def fromFactored(cls, a: sp.Number, x1: sp.Number, x2: sp.Number) -> ParabolaBuilder:
        """
        Creates a parabola builder from a factored function.
        :param a: the a
        :param x1: the x1
        :param x2: the x2
        """
        x = sp.Symbol("x")
        a, b, c = sp.Poly(a * (x - x1) * (x - x2), x).all_coeffs()
        return cls(a, b, c)

    @classmethod
    def fromFactoredExpr(cls, factored: sp.Function) -> ParabolaBuilder:
        """
        Creates a parabola builder from a factored function.
        :param factored: the factored function
        """
        a, b, c = sp.Poly(factored, sp.Symbol("x")).all_coeffs()
        return cls(a, b, c)

    @property
    def a(self) -> sp.Number:
        """a"""
        return self._a

    @property
    def b(self) -> sp.Number:
        """b"""
        return self._b

    @property
    def c(self) -> sp.Number:
        """c"""
        return self._c

    @property
    def u(self) -> sp.Number:
        """u (vertex x coordinate)"""
        return -self._b / (2 * self._a)

    @property
    def v(self) -> sp.Number:
        """v (vertex y coordinate)"""
        return self._c - self._b**2 / (4 * self._a)

    def factoredForm(self, x: sp.Symbol = sp.Symbol("x")) -> sp.Function:
        """
        Returns the factored function.
        :param x: the variable
        """
        rZeros = self.realZeros
        assert len(rZeros) in (1, 2), "cannot factor, not enough zeros"

        if len(zeros) == 1:
            x1 = rZeros.pop()
            return self._a * (x - x1)**2
        x1, x2 = rZeros
        return self._a * (x - x1) * (x - x2)

    def vertexForm(self, x: sp.Symbol = sp.Symbol("x")) -> sp.Function:
        """
        Returns the vertex form.
        """
        return self._a * (x - self.u)**2 + self.v

    def normalForm(self, x: sp.Symbol = sp.Symbol("x")) -> sp.Function:
        """
        Returns the normal form.
        """
        return self._a * x**2 + self.b * x + self.c

    @property
    def vertex(self) -> Tuple[sp.Number, sp.Number]:
        """vertex (u, v)"""
        return self.u, self.v

    @property
    def zeros(self) -> Set[sp.Number]:
        """zeros"""
        return zeros(self.normalForm())

    @property
    def realZeros(self) -> Set[sp.Number]:
        """real zeros"""
        return realZeros(self.normalForm())

    @property
    def up(self) -> bool:
        """open up"""
        return self._a > 0

    @property
    def down(self) -> bool:
        """open down"""
        return self._a < 0

    def __call__(self, x: sp.Symbol = sp.Symbol("x")) -> sp.Function:
        return self.normalForm(x)

    def compress(self, factor: sp.Number) -> ParabolaBuilder:
        """
        Compresses the polynom horizontally by a factor.
        :param factor: the factor
        """
        assert factor != 0
        return ParabolaBuilder.fromVertex(self.a * factor, self.u, self.v)

    def stretch(self, factor: sp.Number) -> ParabolaBuilder:
        """
        Stretches the polynom horizontally by a factor.
        :param factor: the factor
        """
        assert factor != 0
        return ParabolaBuilder.fromVertex(self.a / factor, self.u, self.v)

    def shiftHor(self, factor: sp.Number) -> ParabolaBuilder:
        """
        Shifts the polynom horizontally by a factor.
        :param factor: the factor
        """
        return ParabolaBuilder.fromVertex(self.a, self.u + factor, self.v)

    def shiftVert(self, factor: sp.Number) -> ParabolaBuilder:
        """
        Shifts the polynom vertically by a factor.
        :param factor: the factor
        """
        return ParabolaBuilder.fromVertex(self.a, self.u, self.v + factor)

    def yy(self, xx: np.ndarray) -> np.ndarray:
        """
        Returns the y values for the given x values.
        :param xx: the x values
        """
        f = self()
        fnp = sp.lambdify(sp.Symbol("x"), f, "numpy")
        return fnp(xx)

    def plot(self,
             x: sp.Symbol = sp.Symbol("x"),
             xrange: Tuple[float, float] = (-5, 5),
             yrange: Optional[Tuple[float, float]] = None,
             equalAxis: bool = False) -> mpl.figure.Figure:
        """
        Plots the polynom.
        :param x: the variable
        :param xrange: the x range
        :param yrange: the y range, if None, the y range is automatically determined
        """
        f = self(x)
        xx = np.linspace(*xrange)
        yy = [f.subs(x, i) for i in xx]
        if yrange is None:
            yrange = (min(yy), max(yy))

        fig = plt.figure()
        subplot = fig.add_subplot(111)
        subplot.plot(xx, yy)

        if equalAxis:
            subplot.set_aspect("equal")

        xmin, xmax = xrange
        ymin, ymax = yrange
        subplot.set_xlim(xmin, xmax)
        subplot.set_ylim(float(ymin), float(ymax))
        return fig
