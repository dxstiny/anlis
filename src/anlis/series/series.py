import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def seriesPlot(series: sp.Expr, n: int) -> None:
    """
    Plot the given series.
    :param series: The series to plot.
    :param n: The number of elements to plot.
    """
    # Determine the elements of the series.
    elements = [series.subs("n", i) for i in range(1, n + 1)]

    # Plot the elements of the series.
    plt.plot(elements, "o-")
