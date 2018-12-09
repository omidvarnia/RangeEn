# Simulator of different signals and models
#
# Ref: A. Omidvarnia, M. Mesbah, M. Pedersen, G. Jackson, Range Entropy: a bridge between signal complexity and self-similarity, arxiv, 2018
#
# Written by: Amir Omidvarnia, PhD
# Florey Institute of Neuroscience and Mental Health
# University of Melbourne, Australia
# September 2018
#
# Email: a.omidvarnia@brain.org.au
#
############## Import necessary libraries and set initial parameters
import numpy as np
from Analyses.flm import flm
import nolds
import acoustics.generator as gen
from scipy.integrate import odeint
from scipy.signal import chirp

### Fractional Brownian motion
# Ref: https://pypi.org/project/fbm/0.1.1/
def fBm(N, H):
    # f = FBM(n=N - 1, hurst=H, length=1, method='hosking')
    # x = f.fbm()

    x = nolds.fbm(N,H)
    return x

### Fractional Brownian motion with an extra multiplier D (as per the reviewer's question).
# This implementation is based on the implementation of nolds.fbm at: https://cschoel.github.io/nolds/_modules/nolds/datasets.html#fbm
def fBm_D(N, H, D):
    # f = FBM(n=N - 1, hurst=H, length=1, method='hosking')
    # x = f.fbm()

    assert H > 0 and H < 1 # If this condition is not met, it generates an error message.

    def R(t, s):
        twoH = 2 * H
        return D * 0.5 * (s ** twoH + t ** twoH - np.abs(t - s) ** twoH)

    # form the matrix tau
    gamma = R(*np.mgrid[0:N, 0:N])  # apply R to every element in matrix
    w, P = np.linalg.eigh(gamma)
    L = np.diag(w)
    sigma = np.dot(np.dot(P, np.sqrt(L)), np.linalg.inv(P))
    v = np.random.randn(N)

    x = np.dot(sigma, v)
    return x

### Fractional Levy motion
def fLm(alpha, H, n, dim=1, nm=10):
    # Algorithm to generate fractional Levy motion
    #
    # From Lui et al, A corrected and generalized successive random additions
    # algorithm for simulating Fractional Levy Motion, Mathematical Geology, 36 (2004)
    #
    # Chris Green, 2018
    # chris.green@csiro.au
    #
    # Ref: https://github.com/cpgr/flm

    x = flm(alpha, H, n, dim, nm)

    return x

### White noise
# Ref: https://pypi.org/project/acoustics/
def white_noise(N):
    x = gen.white(N)
    return x

### What noise
# Ref: https://pypi.org/project/acoustics/
def Pink_noise(N):
    x = gen.pink(N)
    return x

### Uniform noise
def Uniform_noise(N, min=0, max=1):
    x = np.random.uniform(min, max, N)
    return x

### What noise
# Ref: https://pypi.org/project/acoustics/
def Blue_noise(N):
    x = gen.blue(N)
    return x

def Sine_wave(N,t1=0,t2=4*np.pi):
    tspan = np.arange(t1, t2, (t2-t1) / N)
    x = np.sin(tspan)
    return x, tspan

### Line
def Line(N):
    x = np.arange(N)
    return x

### Chirp (LFM) signal
def chirp(N,f0=6,f1=1,t1=10,t_end=10):
    # f0: Frequency at time 0
    # t1: Time at which f1 is specified.
    # f1: Frequency (e.g. Hz) of the waveform at time t1.
    # t: Times at which to evaluate the waveform.
    # t_end: Signal length in seconds
    # N: Number of samples

    t = np.linspace(0, t_end, N)
    x = chirp(t, f0, f1, t1, method='linear')
    return x

### The Logistic map
def Logistic_map(N, R=3.8, x0=0.4):
    #
    # x(n+1) = R*x(n)*(1-x(n))
    #
    # N: number of time points
    # R: Logistic map parameter
    # x0: Initial condition
    x = []
    x.append(x0)
    for n in range(0,N-1):
        x.append(R * x[n] * (1 - x[n]))

    return x

### The Henon map
def Henon_map(N,a=1.4,b=0.3,x0=0,y0=0):
    #
    # x(n + 1) = 1 - a * x(n) ^ 2 + y(n)
    #
    # y(n + 1) = b * x(n)
    #
    # N: number of time points
    # a,b : Henon map parameters
    # x0, y0: Initial conditions
    #
    # Ref: M.Henon(1976). "A two-dimensional mapping with a strange attractor".Communications in Mathematical Physics 50(1): 6977
    x = []
    x.append(x0)
    y = []
    y.append(y0)

    for n in range(1, N):
        x.append(1 - a * np.square(x[n - 1]) + y[n - 1])
        y.append(b * x[n - 1])

    return x, y

### Roessler oscillator
def roessler_ode(y,t,omega=1, a=0.35, b=0.2, c=10):
    #
    # x = -omega * z - y
    # y =  x + a * y
    # z = b + z * (x-c)
    #
    dy = np.zeros((3))
    dy[0] = -1.0*(omega*y[1] + y[2]) #+ e1*(y[3]-y[0])
    dy[1] = omega * y[0] + a * y[1]
    dy[2] =  b + y[2] * (y[0] - c)
    return dy

def Roessler_osc(N, t1, t2, omega=1, a=0.35, b=0.2, c=10, x0=1, y0=1, z0=0):
    v = odeint(roessler_ode, [x0, y0, z0], np.arange(t1,t2,(t2-t1)/N),args=(omega,a,b,c))
    x = v[:,0]
    y = v[:,1]
    z = v[:,2]

    return x, y, z

def MIX(N, t1, t2):
    #
    # Ref: Pincus 1991
    #
    Z = np.random.randint(2, size=N) # A uniformly distributed random series of 0's and 1's
    Y = np.random.uniform(0, 1, size=N) # Uniform noise
    X,_,_ = Roessler_osc(N, t1, t2)

    return np.multiply((1-Z),X) + np.std(X) * np.multiply(Z,Y)
