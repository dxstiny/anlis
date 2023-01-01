# ANLIS - Analysis for Python

ANLIS is a Python package for [analysis](https://en.wikipedia.org/wiki/Mathematical_analysis) built on top of [numpy]( https://www.numpy.org/ ) and [sympy]( https://www.sympy.org/en/index.html ). ANLIS provides a set of functions to perform analysis tasks. ANLIS is a work in progress and currently supports the following tasks:

- Series
    - Plotting any series
    - Arithmetic Series
        - Finding $a_n$ from two elements
    - Geometric Series
        - Finding $a_n$ from two elements
        - Finding the sum of an infinite series (based on ratio and first element *or* two elements)
- Convergence
    - Determining if a sequence is convergent (or divergent)
    - Convergence Tests
- Derivatives
    - Finding the critical points of a function
    - Finding the extrema of a function
- Vectors
    - Finding the magnitude of a vector
    - Finding the unit vector of a vector
    - Finding the dot product of two vectors
- Taylor Series
    - Finding the Taylor Series of a function
    - Finding the Taylor Polynomial of a function
    - Finding the Lagrange Remainder of a function
- Integrals
    - Left/Right Riemann Sums
    - Trapezoidal Rule
    - Simpson's Rule
- Differentials (e.g. for error analysis)
    - Absolute differential
    - Relative differential
- Multidimensionals
    - Critical points
        - Finding critical points
        - Finding extrema
    - Derivatives
        - Finding the (or all) partial derivatives of a function
        - Finding partial derivatives of composite functions
        - Finding the gradient of a function
        - Finding the Jacobian of a function
        - Finding the determinant of a function
        - Finding the linearisation of a function
        - Solving with Newton-Raphson
        - Finding the directional derivative of a function
    - Contour Lines
        - Finding the contour lines of a function
        - Finding the tangent lines of contour lines
        - Plotting the contour lines of a function (2D or 3D)
    - Differentials (e.g. for error analysis)
        - Absolute differential
        - Relative differential
        - Total differential

The [wiki](https://github.com/dxstiny/anlis/wiki) contains more information on the functions along with examples and when to use them.
