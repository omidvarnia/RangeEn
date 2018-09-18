# Single-scale analysis of ApEn, SampEn, RangeEn_A and RangeEn_B over the range of Hurst exponents (0 to 1) and tolerance values r
#
# This script generates Fig. 4 of the RangeEn manuscript.
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
############## Import necessary libraries
from os.path import dirname, abspath
import sys
import matplotlib as mpl
import numpy as np
import os, time
import sim_data
import matplotlib.pyplot as plt
import measures

############## Set path
main_folder = dirname(dirname(abspath(__file__))) # /Main RangeEn folder
sys.path.insert(0, main_folder)

############## Set parameters
is_plot  = 1      # Plotting flag (if 1, the final result will be plotted).
force    = 0      # BE CAREFULL! If 1, the analysis will be done, even if there is an existing result file.

N_single = 1000   # Number of time points in the input signal
m_single = 2      # Embeding dimension

r_span   = np.arange(.01, 1, 0.01)               # Span of the tolerance parameter r
r_span   = np.reshape(r_span, (1, len(r_span)))

H_span   = np.arange(0.01, 0.99, 0.01)             # Span of the Hurst exponent H for fractal Brownian motion (fBm)
H_span   = np.reshape(H_span, (1, len(H_span)))

############## Define the results filename
cwd = os.getcwd() # Current working directory
if(not os.path.exists(cwd + os.sep + 'Results')):
    # Create the Results folder, if needed.
    os.mkdir(cwd + os.sep + 'Results')
output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig4_fBm_N' + str(N_single) + '.npz'

############## Perform entropy analysis for fBm over the entire range of Hurst exponents and a range of tolerance values r
_, N_r = np.shape(r_span)   # Number of r values
_, N_H = np.shape(H_span)   # Number of H values

if (not os.path.isfile(output_filename) or force):

    ApEn_h      = np.zeros((N_H, N_r))
    SampEn_h    = np.zeros((N_H, N_r))
    RangeEn_A_h = np.zeros((N_H, N_r))
    RangeEn_B_h = np.zeros((N_H, N_r))

    t00         = time.time()
    for n_h in range(0,N_H):

        H = H_span[0, n_h]

        # Simulate the input signal (fBm)
        x = sim_data.fBm(N_single, H)

        t0 = time.time()
        for n_r in range(0,N_r):

            # Approximate Entropy
            ApEn_h[n_h, n_r]      = measures.ApEn(x, m_single, r_span[0, n_r])

            # Sample Entropy
            SampEn_h[n_h, n_r]    = measures.SampEn(x, m_single, r_span[0, n_r])

            # RangeEn-A (Modified Approximate Entropy)
            RangeEn_A_h[n_h, n_r] = measures.RangeEn_A(x, m_single, r_span[0, n_r])

            # RangeEn-B (Modified Sample Entropy)
            RangeEn_B_h[n_h, n_r] = measures.RangeEn_B(x, m_single, r_span[0, n_r])

            ##### Save the entropy values in an external .npz file
            np.savez(output_filename, SampEn_h=SampEn_h, ApEn_h=ApEn_h, RangeEn_B_h=RangeEn_B_h, RangeEn_A_h=RangeEn_A_h)

        print('Hurst exponent ' + str(r_span[0, n_h]) + ', elapsed time = ' + str(time.time() - t0))

    print('Entire elapsed time = ' + str(time.time()-t00))

else:
    ##### Load the existing output .npz file
    out         = np.load(output_filename)
    SampEn_h    = out['SampEn_h']
    ApEn_h      = out['ApEn_h']
    RangeEn_B_h = out['RangeEn_B_h']
    RangeEn_A_h = out['RangeEn_A_h']

############## Plot the resulting entropy patterns over tolerance r values
if (is_plot):

    #######
    cmap = plt.get_cmap('jet', N_H)
    fig = plt.figure()

    plt.subplot(2,2,1)
    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    ax.autoscale(enable=True, axis='x', tight=True)
    for n_h, n in enumerate(np.linspace(0, 1, N_H)):
        ax.plot(r_span[0,:], ApEn_h[n_h, :], c=cmap(n_h))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.ylim([0,5])
    plt.xlabel('log(r)'), plt.ylabel('ApEn'), plt.title('ApEn')

    plt.subplot(2, 2, 2)
    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    ax.autoscale(enable=True, axis='x', tight=True)
    for n_h, n in enumerate(np.linspace(0, 1, N_H)):
        ax.plot(r_span[0,:], SampEn_h[n_h, :], c=cmap(n_h))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.xlabel('log(r)'), plt.ylabel('SampEn'), plt.title('SampEn')
    plt.colorbar(sm)
    plt.ylim([0, 5])

    plt.subplot(2, 2, 3)
    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    ax.autoscale(enable=True, axis='x', tight=True)
    for n_h, n in enumerate(np.linspace(0, 1, N_H)):
        ax.plot(r_span[0,:], RangeEn_A_h[n_h, :], c=cmap(n_h))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.xlabel('log(r)'), plt.ylabel('RangeEn_A'), plt.title('RangeEn_A')
    plt.colorbar(sm)
    plt.ylim([0, 5])

    plt.subplot(2, 2, 4)
    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    ax.autoscale(enable=True, axis='x', tight=True)
    for n_h, n in enumerate(np.linspace(0, 1, N_H)):
        ax.plot(r_span[0,:], RangeEn_B_h[n_h, :], c=cmap(n_h))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.xlabel('log(r)'), plt.ylabel('RangeEn_B'), plt.title('RangeEn_B')
    plt.colorbar(sm)
    plt.ylim([0, 5])

    plt.show()

print('Finished!')

