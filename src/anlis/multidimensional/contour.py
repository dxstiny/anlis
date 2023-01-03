# -*- coding: utf-8 -*-
"""a toolset for dealing with contour lines of multidimensional functions"""
__copyright__ = ("Copyright (c) 2023 https://github.com/dxstiny")


from typing import List, Union, Tuple, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.plotting.plot import Plot, ContourSeries

from anlis.vector import dotProduct


def contourLines(function: sp.Function,
                 y: sp.Symbol = sp.Symbol('y'),
                 C: Union[float, sp.Symbol] = sp.Symbol('C'),
                 toZero: bool = False,
                 toY: bool = False) -> Union[sp.Eq, List[sp.Eq]]:
    """Returns the contour lines of a function of two variables.
    :param function: function of two variables
    :param y: variable y (if toY is True, y is the variable to solve for)
    :param c: constant c (this variable will be inserted into the function)
    :param toZero: if True, the function is set to zero (c will still be inserted)
    :param toY: if True, the function is set to y (c will still be inserted)
    :return: contour lines of function

    Example:
    >>> from sympy import symbols, sin, cos
    >>> from anlis.multidimensional.contour import contourLines
    >>> x, y, c = symbols('x y c')

    >>> contourLines(x**2 + y**2)
    Eq(x**2 + y**2, c)

    >>> contourLines(x**2 + y**2, y, c)
    Eq(x**2 + y**2, c)

    >>> contourLines(x**2 + y**2, toZero=True)
    Eq(x**2 + y**2 - c, 0)

    >>> contourLines(x**2 + y**2, toY=True)
    [Eq(y, -sqrt(c - x**2)), Eq(y, sqrt(c - x**2))]
    """
    if toZero:
        return sp.Eq(function - C, 0)
    if toY:
        sol = sp.solve(function - C, y)
        return [ sp.Eq(y, s) for s in sol ]
    return sp.Eq(function, C)

def contourPlotSp(function: sp.Function,
                  x: sp.Symbol = sp.Symbol('x'),
                  y: sp.Symbol = sp.Symbol('y'),
                  xmin: int = -10,
                  xmax: int = 10,
                  ymin: int = -10,
                  ymax: int = 10) -> Plot:
    """Returns the contour plot of a function of two variables using SymPy.
    :param function: function of two variables
    :param x: variable x
    :param y: variable y
    :param xmin: minimum value of x
    :param xmax: maximum value of x
    :param ymin: minimum value of y
    :param ymax: maximum value of y
    :return: contour plot of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.contour import contourPlotSp
    >>> x, y = symbols('x y')
    >>> contourPlotSp(x**2 + y**2).show()
    """
    return Plot(ContourSeries(function, (x, xmin, xmax), (y, ymin, ymax)))

def contourPlot(function: sp.Function,
                x: sp.Symbol = sp.Symbol('x'),
                y: sp.Symbol = sp.Symbol('y'),
                xmin: int = -10,
                xmax: int = 10,
                ymin: int = -10,
                ymax: int = 10,
                count: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the contour plot of a function of two variables using NumPy.
    :param function: function of two variables
    :param x: variable x
    :param y: variable y
    :param xmin: minimum value of x
    :param xmax: maximum value of x
    :param ymin: minimum value of y
    :param ymax: maximum value of y
    :param count: number of contour lines
    :return: contour plot of function

    Example:
    >>> import matplotlib.pyplot as plt
    >>> from sympy import symbols
    >>> from anlis.multidimensional.contour import contourPlot
    >>> x, y = symbols('x y')
    >>> X, Y, Z = contourPlot(x**2 + y**2)
    >>> plt.contour(X, Y, Z, levels = 10) # levels sets the number of contour lines
    >>> plt.show()
    """
    if count < 2:
        raise ValueError('Count must be greater than 1.')
    if count < 10:
        print('Warning: count is less than 10 - the contour plot may not be smooth.')

    xx = np.linspace(xmin, xmax, count)
    yy = np.linspace(ymin, ymax, count)
    X, Y = np.meshgrid(xx, yy)
    Z = sp.lambdify((x, y), function, 'numpy')(X, Y)
    return X, Y, Z

def contourTangent(gradient: Tuple[sp.Function],
                   x0: Tuple[float],
                   variables: Tuple[sp.Symbol]) -> sp.Function:
    """Returns the tangent to a contour line.
    :param gradient: gradient of the function
    :param x0: point on the function
    :param variables: variables of the tangent
    :return: tangent to the contour line
    """
    assert len(gradient) == len(x0) == len(variables)

    gx0 = [ g.subs({variables[i]: x0[i] for i in range(len(variables))}) for g in gradient ]
    dx = [ variables[i] - x0[i] for i in range(len(variables)) ]
    return dotProduct(tuple(gx0), dx)


class View:
    """Represents a 3D view."""
    def __init__(self, elevation: float = 30, azimuth: float = 240) -> None:
        """Creates a 3D view.
        :param elevation: elevation of the view
        :param azimuth: azimuth of the view
        """
        self.elevation = elevation
        self.azimuth = azimuth

    @staticmethod
    def top() -> 'View':
        """
        Returns the top view.

        y-axis is pointing to the right,
        x-axis is pointing to the bottom
        """
        return View(90, 0)

    @staticmethod
    def bottom() -> 'View':
        """
        Returns the bottom view.

        y-axis is pointing to the right,
        x-axis is pointing to the top
        """
        return View(270, 0)

    @staticmethod
    def front() -> 'View':
        """
        Returns the front view.

        y-axis is pointing to the right,
        z-axis is pointing to the top
        """
        return View(0, 0)

    @staticmethod
    def back() -> 'View':
        """
        Returns the back view.

        y-axis is pointing to the left,
        z-axis is pointing to the top
        """
        return View(0, 180)

    @staticmethod
    def left() -> 'View':
        """
        Returns the left view.

        x-axis is pointing to the right,
        z-axis is pointing to the top
        """
        return View(0, 270)

    @staticmethod
    def right() -> 'View':
        """
        Returns the right view.

        x-axis is pointing to the left,
        z-axis is pointing to the top
        """
        return View(0, 90)

    @staticmethod
    def isometric() -> 'View':
        """Returns the isometric view."""
        return View(30, 240)

def contourPlot3d(function: sp.Function, # pylint: disable=too-many-locals
                  x: sp.Symbol = sp.Symbol('x'),
                  y: sp.Symbol = sp.Symbol('y'),
                  z: sp.Symbol = sp.Symbol('z'),
                  xrange: Tuple[int] = (-10, 10),
                  yrange: Tuple[int] = (-10, 10),
                  zrange: Optional[Tuple[int]] = None,
                  view: View = View.isometric(),
                  count: int = 100) -> mpl.figure.Figure:
    """Plots the contour plot of a function of two variables using Matplotlib.
    :param function: function of two variables
    :param x: variable x
    :param y: variable y
    :param z: variable z
    :param xrange: range of x
    :param yrange: range of y
    :param zrange: range of z, if None, the range is automatically determined by min and max values
    :param view: view of the plot
    :param count: number of contour lines
    :return: None

    This function is based on a python notebook by Joachim Wirth.
    While the original implementation offered  more customisation, this function is easier to use.
    It also uses SymPy in favour of NumPy, which makes it a bit more straightforward to use.

    Example:
    >>> import sympy as sp
    >>> from anlis.multidimensional.contour import contourPlot3d, View
    >>> x, y = sp.symbols('x y')
    >>> f = 2*x**2 - 3*x*y + 8*y**2 + x - y
    >>> contourPlot3d(f, x, y)
    >>> contourPlot3d(f, x, y, view = View.top())
    >>> contourPlot3d(f, x, y, xrange = (-5, 5), yrange = (-5, 5), count = 50)
    >>> contourPlot3d(f, x, y, zrange = (-10, 10))
    """
    if count < 2:
        raise ValueError('Count must be greater than 1.')
    if count < 10:
        print('Warning: count is less than 10 - the contour plot may not be smooth.')

    xmin, xmax = xrange
    ymin, ymax = yrange

    xx = np.linspace(xmin, xmax, count)
    yy = np.linspace(ymin, ymax, count)

    X, Y = np.meshgrid(xx, yy)
    Z = sp.lambdify((x, y), function, 'numpy')(X, Y)

    if zrange is None:
        zmin, zmax = Z.min(), Z.max()
    else:
        zmin, zmax = zrange

    zz = np.linspace(zmin, zmax, count)

    fig = plt.figure()
    plot = fig.add_subplot(projection='3d')
    plot.set_xlabel(x.name)
    plot.set_ylabel(y.name)
    plot.set_zlabel(z.name)
    plot.set_xlim(xmin, xmax)
    plot.set_ylim(ymin, ymax)
    plot.set_zlim(zmin, zmax)
    plot.view_init(elev = view.elevation, azim = view.azimuth)

    plot.contour(X, Y, Z, zz, colors = ["blue"], linewidths = [0.5])
