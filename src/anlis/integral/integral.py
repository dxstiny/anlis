import numpy as np

from anlis.types import Function


def RiemannSum(f: Function,
               a: float,
               b: float,
               n: int,
               upper: bool = False) -> float:
    """
    Riemann sum for approximating the integral of f(x) in [a, b].
    :param f: function f(x)
    :param a: left endpoint of interval
    :param b: right endpoint of interval
    :param n: number of subintervals
    :param upper: if True, use upper Riemann sum
    :return: approximation of integral
    """
    h = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)
    if upper:
        return h * sum(y[1:])
    return h * sum(y[:-1])

def TrapezoidalRule(f: Function,
                    a: float,
                    b: float,
                    n: int) -> float:
    """
    Trapezoidal rule for approximating the integral of f(x) in [a, b].
    :param f: function f(x)
    :param a: left endpoint of interval
    :param b: right endpoint of interval
    :param n: number of subintervals
    :return: approximation of integral
    """
    h = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/2 * (y[0] + 2*sum(y[1:-1]) + y[-1])

def SimpsonsRule(f: Function,
                 a: float,
                 b: float,
                 n: int) -> float:
    """
    Simpson's rule for approximating the integral of f(x) in [a, b].
    :param f: function f(x)
    :param a: left endpoint of interval
    :param b: right endpoint of interval
    :param n: number of subintervals
    :return: approximation of integral
    """
    h = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/3 * (y[0] + 4*sum(y[1:-1:2]) + 2*sum(y[2:-1:2]) + y[-1])
