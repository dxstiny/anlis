from typing import Tuple, List, Optional
import sympy as sp

from anlis.vector import unitVector, dotProduct


def partialDerivative(function: sp.Function, variable: sp.Symbol) -> sp.Function:
    """Returns the partial derivative of a function of two variables.
    :param function: function of two variables
    :param variable: variable
    :return: partial derivative of function with respect to variable

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.derivative import partialDerivative
    >>> x, y = symbols('x y')
    >>> partialDerivative(x**2 + y**2, x)
    2*x
    >>> partialDerivative(x**2 + y**2, y)
    2*y
    """
    return sp.diff(function, variable).simplify()

def partialDerivatives(function: sp.Function,
                       *variables: sp.Symbol) -> Tuple[sp.Function, ...]:
    """Returns the partial derivatives of a function of any number of variables.
    :param function: function of two or more variables
    :param variables: variables
    :return: partial derivatives of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.derivative import partialDerivatives
    >>> x, y = symbols('x y')
    >>> partialDerivatives(x**2 + y**2, x, y)
    (2*x, 2*y)
    """
    return tuple(partialDerivative(function, v) for v in variables)

def gradient(function: sp.Function,
             *variables: sp.Symbol) -> Tuple[sp.Function, ...]:
    """Returns the gradient of a function of any number of variables.
    :param function: function of two or more variables
    :param variables: variables
    :return: gradient of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.derivative import gradient
    >>> x, y = symbols('x y')
    >>> gradient(x**2 + y**2, x, y)
    (2*x, 2*y)
    """
    return tuple(partialDerivative(function, v) for v in variables)

def linearise(function: sp.Function,
              x0: Tuple[float, ...],
              *variables: sp.Symbol) -> sp.Function:
    """Returns the linearisation of a function of any number of variables.
    :param function: function of two or more variables
    :param variables: variables
    :return: linearisation of function

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.derivative import linearise
    >>> x, y = symbols('x y')
    >>> linearise(x**2 + y**2, x, y)
    """
    assert len(x0) == len(variables)
    f0 = function.subs({ v: x for v, x in zip(variables, x0) })
    total = f0
    for v, x in zip(variables, x0):
        total += partialDerivative(function, v).subs({ v: x }) * (v - x)
    return total

def jacobiMatrix(functions: List[sp.Function],
                 variables: List[sp.Symbol]) -> sp.Matrix:
    """Returns the Jacobi matrix of a function of any number of variables.
    :param functions: functions of two or more variables
    :param variables: variables
    :return: Jacobi matrix of functions

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.derivative import JacobiMatrix
    >>> x, y = symbols('x y')
    >>> JacobiMatrix(x**2 + y**2, x, y)
    Matrix([
    [2*x, 2*y]])
    """
    assert len(functions) == len(variables)
    return sp.Matrix([ gradient(f, *variables) for f in functions ])

def determinant(function: sp.Function,
                variables: List[sp.Symbol]) -> sp.Function:
    """Returns the determinant of the Jacobi matrix of a function of any number of variables.
    :param functions: functions of two or more variables
    :param variables: variables
    :return: determinant of Jacobi matrix of functions

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.derivative import determinant
    >>> x, y = symbols('x y')
    >>> determinant(x**2 + y**2, x, y)
    4*x**2
    """
    return partialDerivative2(function, *variables).det()

def newtonRaphson(functions: Tuple[sp.Function],
                  variables: Tuple[sp.Symbol],
                  x0: Tuple[float],
                  n: int = 10) -> Tuple[float, ...]:
    """Returns the solution of a system of equations using Newton-Raphson.
    it stops after n iterations.
    :param functions: functions of two or more variables
    :param variables: variables
    :param x0: initial guess
    :param n: number of iterations
    :return: solution

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.derivative import newtonRaphson
    >>> x, y = symbols('x y')
    >>> newtonRaphson([x1**2 + x2**2 - 1, x1 - x2], [x1, x2], [1, 2], 2)
    (3/4, 3/4)
    """
    assert len(functions) == len(variables)
    assert len(x0) == len(variables)
    x = sp.Matrix(x0)
    J = jacobiMatrix(functions, variables)

    for _ in range(n):
        f = sp.Matrix([ f.subs({ v: x for v, x in zip(variables, list(x)) })
                        for f in functions ])
        x = x - J.inv().subs({ v: x for v, x in zip(variables, list(x)) }) * f
    return tuple(x)

def newtonRaphsonWithEpsilon(functions: Tuple[sp.Function],
                             variables: Tuple[sp.Symbol],
                             x0: Tuple[float],
                             epsilon: float = 1e-6,
                             maxIterations: int = 1000) -> Tuple[int, Tuple[float, ...]]:
    """Returns the solution of a system of equations using Newton-Raphson.
    it stops when the error is less than epsilon or when the maximum number of iterations is reached.
    :param functions: functions of two or more variables
    :param variables: variables
    :param x0: initial guess
    :param epsilon: maximum error
    :param maxIterations: maximum number of iterations
    :return: solution

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.derivative import newtonRaphsonWithEpsilon
    >>> x, y = symbols('x y')
    >>> newtonRaphsonWithEpsilon([x1**2 + x2**2 - 1, x1 - x2], [x1, x2], [1, 2], 0.0001)
    (2, (0.707106781186547, 0.707106781186547))
    """
    assert len(functions) == len(variables)
    assert len(x0) == len(variables)
    x = sp.Matrix(x0)
    J = jacobiMatrix(functions, variables)

    for i in range(maxIterations):
        f = sp.Matrix([ f.subs({ v: x for v, x in zip(variables, list(x)) })
                        for f in functions ])
        if f.norm() < epsilon:
            break
        x = x - J.inv().subs({ v: x for v, x in zip(variables, list(x)) }) * f
    return i, tuple(x)

def partialCompositeDerivative(z: sp.Function,
                               functions: Tuple[sp.Eq],
                               symbol: sp.Symbol) -> sp.Function:
    """Returns the partial composite derivative of a function of any number of variables.
    :param z: function of two or more variables
    :param functions: equations
    :param symbol: variable
    :return: partial composite derivative of z

    Example:
    given: z = f(x, y) = x**2 * e**y, x = 4u, y = 3u**2 - 2v
    find: dz/du
    >>> from sympy import symbols, Eq
    >>> from anlis.multidimensional.derivative import partialCompositeDerivative
    >>> u, v = symbols('u v')
    >>> x, y = symbols('x y')
    >>> z = x**2 * exp(y)
    >>> partialCompositeDerivative(z, [Eq(x, 4*u), Eq(y, 3*u**2 - 2*v)], u)
    32*u*(3*u**2 + 1)*exp(3*u**2 - 2*v)
    """

    def _findFunction(symbol: sp.Symbol) -> Optional[sp.Function]:
        # if the symbol is on the left side of the equation, return the right side
        for function in functions:
            if function.lhs == symbol:
                return function.rhs
        return None

    # recursively compose a final function
    def _composeFunction(function: sp.Function) -> sp.Function:

        for variable in function.free_symbols:
            f = _findFunction(variable)
            if not f: # just a symbol
                continue
            function = function.subs({ variable: _composeFunction(f) })
        return function


    print(_composeFunction(z))
    return partialDerivative(_composeFunction(z), symbol)


def partialDerivative2(function: sp.Function,
                       *variables: sp.Symbol) -> sp.Matrix:
    """Returns the second partial derivatives of a function of two variables. (fxx, fxy, fyx, fyy)
    :param function: function of two or more variables
    :param variables: variables
    :return: partial derivative matrix

    Example:
    >>> from sympy import symbols
    >>> from anlis.multidimensional.derivative import partialDerivative2
    >>> x, y = symbols('x y')
    >>> partialDerivative2(x**2 + y**2, [x, y])
    Matrix([
        [2, 0],
        [0, 2]])
    """
    assert len(variables) == 2

    f1 = partialDerivatives(function, *variables)
    f2 = [ partialDerivatives(f, *variables) for f in f1 ]
    return sp.Matrix(f2)

def partialDerivativeN(f: sp.Function,
                       n: Tuple[sp.Symbol]) -> sp.Function:
    """Returns the partial derivative of a function of two or more variables.
    :param f: function of two or more variables
    :param n: variables in the order of the partial derivative (f_[xx], f_[xy], f_[xyz], ...)
    :return: partial derivative
    """
    g = f
    for i in n:
        g = partialDerivative(g, i)
    return g

def directionalDerivative(f: sp.Function,
                          x0: Tuple[float],
                          e: Tuple[float],
                          *variables: sp.Symbol) -> float:
    """Returns the directional derivative of a function of two or more variables.
    :param f: function of two or more variables
    :param x0: initial point
    :param e: direction vector (NOT the unit vector)
    :return: directional derivative
    """
    assert len(variables) == len(x0)
    assert len(variables) == len(e)
    g = gradient(f, *variables)
    g0 = [ g.subs({ v: x for v, x in zip(variables, x0) }) for g in g ]
    ev = unitVector(*e)
    return dotProduct(g0, ev)
