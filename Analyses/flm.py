# Algorithm to generate fractional Levy motion
#
# From Lui et al, A corrected and generalized successive random additions
# algorithm for simulating Fractional Levy Motion, Mathematical Geology, 36 (2004)
#
# Chris Green, 2018
# chris.green@csiro.au

import numpy as np
from Analyses import levy


def truncatedLevy(alpha, max):
    """
    Generate a truncated Levy stable random number
    Ensures that the absolute value of the Levy stable
    random number is less than max
    Parameters
    ----------
    alpha : float
        Levy stable alpha parameter
    max : float
        Maximum of absolute value of Levy random number
    Returns
    -------
    float
        Random number sampled from the Levy stable distribution
    """
    v = levy.levy(alpha)

    while np.abs(v) > max:
        v = levy.levy(alpha)

    return v

def delta(alpha, H, n):
    """
    Calculates the width delta (from Liu et al, 2004) for a given
    level n
    Parameters
    ----------
    alpha : float
        Levy stable alpha parameter
    H : float
        Hurst parameter
    n : int
        level
    Returns
    -------
    float
        Width of the Levy stable random number
    """
    return (1 - np.power(2.0, alpha * H) / np.power(2.0, alpha)) / np.power(2.0, n * alpha * H)

def addLevy(array, alpha, d, max):
    """
    Add a random Levy number to each element in an array
    Adds a random number drawn from the Levy stable distribution
    to each element in a 1, 2 or 3D numpy array.
    Parameters
    ----------
    array : numpy array
        1, 2 or 3D numpy array
    alpha : float
        Levy stable alpha parameter
    d : float
        Width delta
    max : float
        Maximum of absolute value of Levy random number
    Returns
    -------
    float
        Modifies array in place
    """
    lvy = [d * np.power(0.5, 1.0 / alpha) * truncatedLevy(alpha, max) for i in range(array.size)]
    lvy = np.asarray(lvy).reshape(array.shape)
    array += lvy

def interpolateArray(array):
    """
    Adds midpoints in an array by linear interpolation
    Takes an array of size n and returns an array of size
    2n-1 by adding midpoints into the arrayself. Similarly
    for 2D and 3D arrays
    Parameters
    ----------
    array : numpy array
        1, 2 or 3D numpy array
    Returns
    -------
    array
        numpy array with midpoints
    """
    dim = int(array.ndim)
    n = int(np.rint(np.power(array.size, 1.0 / float(dim))))

    orig = array
    array = np.zeros(shape = [2 * n - 1] * dim)

    if dim == 1:
        midpoints = orig[:-1] + np.diff(orig) / 2.0
        array[::2] = orig
        array[1::2] = midpoints

    elif dim == 2:
        # Do axis 1 (the rows)
        midpoints = orig[:, :-1] + np.diff(orig) / 2.0
        array[::2, ::2] = orig
        array[::2, 1::2] = midpoints

        # Do axis 0 (the columns)
        array[1::2] = array[::2][:-1] + np.diff(array[::2], axis=0) / 2.0

    elif dim == 3:
        # Do axis 2 (the rows)
        midpoints = orig[:, :, :-1] + np.diff(orig) / 2.0
        array[::2, ::2, ::2] = orig
        array[::2, ::2, 1::2] = midpoints

        # Do axis 0 (the columns)
        array[1::2] = array[::2][:-1] + np.diff(array[::2], axis=0) / 2.0

        # Do axis 1 (the columns)
        array[:, 1::2] = array[:, ::2][:, :-1] + np.diff(array[:, ::2], axis=1) / 2.0

    else:
        raise ValueError('The dimension of array must be 1, 2 or 3')

    return array

def flm(alpha, H, n, dim=1, nm=50, max=10, progress=False):
    """
    Generates an array of fractional Levy motion of size 2**n + 1
    in each dimesion
    Parameters
    ----------
    alpha : float
        Levy stable alpha parameter
    H : float
        Hurst parameter
    n : int
        Number of points to generate: 2**n + 1
    dim : int
        Dimension (1, 2, or 3). Default is 1
    nm: int
        Number of additional random Levy numbers to add. Default is 50
    max: float
        Maximum absolute value of random number sampled. Default is 10
    progress:  bool
        Whether to print out an indication of progress. Default is false
    Returns
    -------
    array
        Array of flm data points
    """
    if dim < 1 or dim > 3:
        raise ValueError('dim must be 1, 2 or 3')

    # Begin by adding L(alpha) to the endpoints/cornerpoints
    array = np.zeros(shape=[2] * dim)
    addLevy(array, alpha, 1, max)

    # Increase the length of the array by interpolating to add midpoints
    for i in range(0,n):
        array = interpolateArray(array)

        # Add random Levy numbers with width delta
        d = delta(alpha, H, i)
        addLevy(array, alpha, d, max)

        if progress:
            print("Generated level " + str(i + 1))

    # Add more random variables
    for k in range(n, nm):
        d = delta(alpha, H, k)
        addLevy(array, alpha, d, max)

        if progress:
            print("Adding additional random numbers " + str(k + 1))

    return array