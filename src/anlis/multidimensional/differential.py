import sympy as sp
from typing import Tuple


def totalDifferential(function: sp.Function,
                      differentials: Tuple[sp.Symbol],
                      *variables: sp.Symbol) -> sp.Function:
    """Returns the total differential of a function of any number of variables.
    :param function: function of two or more variables
    :param differentials: differentials
    :param variables: variables
    :return: total differential of function
    """
    assert len(differentials) == len(variables)
    return sum( sp.diff(function, v) * d for v, d in zip(variables, differentials) )

def absoluteDifferential(function: sp.Function,
                         x0: Tuple[float, ...],
                         dx: Tuple[float, ...],
                         *variables: sp.Symbol) -> sp.Function:
    """Returns the absolute differential of a function w/ multiple variables
    :param function: function
    :param variables: variables
    :param x0: point
    :param dx: absolute differential of x0
    :return: absolute differential of function
    """
    assert len(x0) == len(variables)
    assert len(dx) == len(variables)

    derivatives = [ sp.diff(function, v) for v in variables ]
    return sum( d.subs(zip(variables, x0)) * e for d, e in zip(derivatives, dx) )

def relativeDifferential(function: sp.Function,
                         x0: Tuple[float, ...],
                         dx: Tuple[float, ...],
                         *variables: sp.Symbol) -> sp.Function:
    """Returns the relative differential of a function w/ multiple variables
    :param function: function
    :param variables: variables
    :param x0: point
    :param dx: relative differential of x0
    :return: relative differential of function
    """
    return absoluteDifferential(function, x0, dx, *variables) / function.subs(zip(variables, x0))
