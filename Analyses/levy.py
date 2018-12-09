# Algorithm to calculate Levy stable random numbers
#
# From Mantegna, Fast, accurate algorithm for numerical simulation of Levy stable
# stochastic processes, Phys Rev E, 49 (1994)
#
# Chris Green, 2018
# chris.green@csiro.au

import numpy as np
from scipy.special import gamma as gamma

# The function sigmax(alpha) - Eq. 12 from Mantegna (1994)
def sigmax(alpha):
    numerator = gamma(alpha + 1.0) * np.sin(np.pi * alpha / 2.0)
    denominator = gamma((alpha + 1)/2.0) * alpha * np.power(2.0, (alpha - 1.0) / 2.0)

    return np.power(numerator / denominator, 1.0 / alpha)

# The function K(alpha) - Eq. 20 from Mantegna (1994)
def K(alpha):
    k = alpha * gamma((alpha + 1.0)/(2.0 * alpha))/ gamma(1.0 / alpha)
    k *= np.power(alpha * gamma((alpha + 1.0)/2.0) / (gamma(alpha + 1.0) * np.sin(np.pi * alpha / 2.0)), 1.0 / alpha)

    return k

# The function C(alpha). Note that C is found by linear interpolation of the points given in Mantegna (1994)
def C(alpha):
    x = np.array((0.75, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.95, 1.99))
    y = np.array((2.2085, 2.483, 2.7675, 2.945, 2.941, 2.9005, 2.8315, 2.737, 2.6125, 2.4465, 2.206, 1.7915, 1.3925, 0.6089))

    return np.interp(alpha, x, y)

# The stochastic variable v(alpha) (Eq. 6 from Mantegna, 1994)
def vf(alpha):
    # Sample two random number from the normal distribution N(0,1)
    x = np.random.normal(0, 1)
    y = np.random.normal(0, 1)

    # Scale x by sigmax(alpha) so that it's standard deviation is sigmax
    x = x * sigmax(alpha)

    # The random variable v
    return x / np.power(np.abs(y), 1.0 / alpha)

# Calculates a single Levy stable random number with alpha and gamma
def levy(alpha, gamma=1, n=1):
    # The Levy random number is found by a weighted average of n independent
    # random variables
    w = 0;
    for i in range(0,n):
        v = vf(alpha)

        # To avoid possible overflow in exp(-v/C) if v is a large negative number,
        # get another v
        while v < -10:
            v = vf(alpha)

        # Transform random variable (Eq. 15 from Mantegna, 1994)
        w += v * ((K(alpha) - 1.0) * np.exp(-v / C(alpha)) + 1.0)

    # The Levy random variable is then
    z = 1.0 / np.power(n, 1.0 / alpha) * w * gamma

    return z