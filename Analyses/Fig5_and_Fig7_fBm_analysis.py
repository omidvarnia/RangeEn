# Entropy analysis of fractional Brownian motion (fBm) over the range of Hurst exponents (0 to 1) and tolerance values r
#
# This script generates Fig. 5 and Fig. 7 of the below manuscript:
#
# If the flag 'force' is 1, the code is executed from scratch and may take some time to be finished.
# If the flag 'force' is 0, the code loads pre-saved results in the 'Results' folder.
# Choose 'STD_correction' between 'yes' and 'no' to generate results with and without amplitude correction, respectively.
#
# A. Omidvarnia, M. Mesbah, M. Pedersen, G. Jackson, Range Entropy: a bridge between signal complexity and self-similarity, Entropy, 2018
#
# Written by: Amir Omidvarnia, PhD
# Florey Institute of Neuroscience and Mental Health
# University of Melbourne, Australia
# December 2018
#
# Email: a.omidvarnia@brain.org.au
#
############## Import necessary libraries
from os.path import dirname, abspath
import sys
import matplotlib as mpl
import numpy as np
import os, time
import matplotlib.pyplot as plt


############## Set path
main_folder = dirname(dirname(abspath(__file__))) # /Main RangeEn folder
sys.path.insert(0, main_folder)

from Analyses import sim_data, measures

############## Set parameters
is_plot  = 1      # Plotting flag (if 1, the final result will be plotted).
force    = 0      # BE CAREFULL! If 1, the analysis will be done, even if there is an existing result file.

N_single = 1000   # Number of time points in the input signal
m_single = 2      # Embeding dimension

r_span   = np.arange(.01, 1, 0.01)               # Span of the tolerance parameter r
r_span   = np.reshape(r_span, (1, len(r_span)))

H_span   = np.arange(0.01, 0.99, 0.01)             # Span of the Hurst exponent H for fractal Brownian motion (fBm)
H_span   = np.reshape(H_span, (1, len(H_span)))

STD_correction = 'no' # Correction of the signal amplitude for ApEn and SampEn through dividing the input signal by its STD: 'yes' or 'no'

############## Define the results filename
cwd = os.getcwd() # Current working directory
if(not os.path.exists(cwd + os.sep + 'Results')):
    # Create the Results folder, if needed.
    os.mkdir(cwd + os.sep + 'Results')

if(STD_correction=='yes'):
    output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig4_fBm_N' + str(N_single) + '_STDcorrection.npz'
else:
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
        if (STD_correction == 'yes'):
            x = x / np.std(x)

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

    plt.suptitle('Fractional Brownian motion - STD correction: ' + STD_correction)

    ############################################################
    ### Plot the relationship between entropy exponents in the m-domain and the Hurst exponent with and withour amplitude correction
    plt.figure()
    p_entropy_hurst = np.zeros((4, 2))  # Slopes of the fitted lines in the r-hurst plane (slope of r-trajectories versus hurst exponent)
    plt.draw()

    for n_filename in range(0, 2):
        if (n_filename == 0):
            output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig4_fBm_N' + str(N_single) + '_STDcorrection.npz'
            marker_color = '*k'
            line_color = '-k'
        else:
            output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig4_fBm_N' + str(N_single) + '.npz'
            marker_color = '*r'
            line_color = '-r'

        out = np.load(output_filename)
        ApEn_h = out['ApEn_h']
        SampEn_h = out['SampEn_h']
        RangeEn_A_h = out['RangeEn_A_h']
        RangeEn_B_h = out['RangeEn_B_h']

        ############## Remove NaN and Info values from the entropy measures
        p_ApEn = np.zeros((1, N_H))
        p_SampEn = np.zeros((1, N_H))
        p_RangeEn_A = np.zeros((1, N_H))
        p_RangeEn_B = np.zeros((1, N_H))

        ### Extract the entropy exponents (fit a line on log10(entropy(m)) versus log10(m) at each H
        for n_h in range(0, N_H):
            # ApEn
            undef = np.isnan(ApEn_h[n_h, :]) + np.isinf(ApEn_h[n_h, :])
            ind1 = np.array([i for i, x in enumerate(undef.flatten()) if x == False])
            if (len(ind1) == N_r):
                tmp = np.polyfit(np.log10(r_span[0, :]), (ApEn_h[n_h, :]), 1)
                p_ApEn[0, n_h] = tmp[0]
            else:
                p_ApEn[0, n_h] = np.nan

            # SampEn
            undef = np.isnan(SampEn_h[n_h, :]) + np.isinf(SampEn_h[n_h, :])
            ind2 = np.array([i for i, x in enumerate(undef.flatten()) if x == False])
            if (len(ind2) == N_r):
                tmp = np.polyfit(np.log10(r_span[0, :]), (SampEn_h[n_h, :]), 1)
                p_SampEn[0, n_h] = tmp[0]
            else:
                p_SampEn[0, n_h] = np.nan

            # RangeEn_A
            undef = np.isnan(RangeEn_A_h[n_h, :]) + np.isinf(RangeEn_A_h[n_h, :])
            ind3 = np.array([i for i, x in enumerate(undef.flatten()) if x == False])
            if (len(ind3) == N_r):
                tmp = np.polyfit(np.log10(r_span[0, :]), (RangeEn_A_h[n_h, :]), 1)
                p_RangeEn_A[0, n_h] = tmp[0]
            else:
                p_RangeEn_A[0, n_h] = np.nan

            # RangeEn_B
            undef = np.isnan(RangeEn_B_h[n_h, :]) + np.isinf(RangeEn_B_h[n_h, :])
            ind4 = np.array([i for i, x in enumerate(undef.flatten()) if x == False])
            if (len(ind4) == N_r):
                tmp = np.polyfit(np.log10(r_span[0, :]), (RangeEn_B_h[n_h, :]), 1)
                p_RangeEn_B[0, n_h] = tmp[0]
            else:
                p_RangeEn_B[0, n_h] = np.nan

        ########################### Plot Entropy curves versus dimension
        ###  Fit a line over the extracted entropy exponents
        undef = np.isnan(p_ApEn[0, :]) + np.isinf(p_ApEn[0, :])
        ind1 = np.array([i for i, x in enumerate(undef.flatten()) if x == False])
        tmp = np.polyfit(H_span[0, ind1], p_ApEn[0, ind1], 1)
        p_entropy_hurst[0, n_filename] = tmp[0]
        plt.subplot(221), plt.plot(H_span[0, :], p_ApEn[0, :], marker_color, markersize=6)
        plt.plot(H_span[0, :], tmp[0] * H_span[0, :] + tmp[1], line_color)
        plt.xlim(0, 1)
        plt.xlabel('Hurst exponent H')
        plt.ylabel('Entropy exponent in the r-plane')
        plt.title('ApEn')
        plt.legend(['With correction', 'With correction, p: ' + '%0.2f' % (p_entropy_hurst[0, 0]), 'Withou correction',
                    'Without correction, p: ' + '%0.2f' % (p_entropy_hurst[0, 1])])

        ###  Fit a line over the extracted entropy exponents
        undef = np.isnan(p_SampEn[0, :]) + np.isinf(p_SampEn[0, :])
        ind2 = np.array([i for i, x in enumerate(undef.flatten()) if x == False])
        if (len(ind2) > 0):
            tmp = np.polyfit(H_span[0, ind2], p_SampEn[0, ind2], 1)
            p_entropy_hurst[1, n_filename] = tmp[0]
            plt.subplot(222), plt.plot(H_span[0, :], p_SampEn[0, :], marker_color, markersize=6)
            plt.plot(H_span[0, :], tmp[0] * H_span[0, :] + tmp[1], line_color)
            plt.xlim(0, 1)
            plt.xlabel('Hurst exponent H')
            plt.ylabel('Entropy exponent in the r-plane')
            plt.title('SampEn')
            plt.legend(['With correction', 'With correction, p: ' + '%0.2f' % (p_entropy_hurst[1, 0]), 'Withou correction',
                        'Without correction, p: ' + '%0.2f' % (p_entropy_hurst[1, 1])])

        ###  Fit a line over the extracted entropy exponents
        undef = np.isnan(p_RangeEn_A[0, :]) + np.isinf(p_RangeEn_A[0, :])
        ind3 = np.array([i for i, x in enumerate(undef.flatten()) if x == False])
        tmp = np.polyfit(H_span[0, ind3], p_RangeEn_A[0, ind3], 1)
        p_entropy_hurst[2, n_filename] = tmp[0]
        plt.subplot(223), plt.plot(H_span[0, :], p_RangeEn_A[0, :], marker_color, markersize=6)
        plt.plot(H_span[0, :], tmp[0] * H_span[0, :] + tmp[1], line_color)
        plt.xlim(0, 1)
        plt.xlabel('Hurst exponent H')
        plt.ylabel('Entropy exponent in the r-plane')
        plt.title('RangeEn-A')
        plt.legend(['With correction', 'With correction, p: ' + '%0.2f' % (p_entropy_hurst[2, 0]), 'Withou correction',
                    'Without correction, p: ' + '%0.2f' % (p_entropy_hurst[2, 1])])

        ###  Fit a line over the extracted entropy exponents
        undef = np.isnan(p_RangeEn_B[0, :]) + np.isinf(p_RangeEn_B[0, :])
        ind4 = np.array([i for i, x in enumerate(undef.flatten()) if x == False])
        tmp = np.polyfit(H_span[0, ind4], p_RangeEn_B[0, ind4], 1)
        p_entropy_hurst[3, n_filename] = tmp[0]
        plt.subplot(224), plt.plot(H_span[0, :], p_RangeEn_B[0, :], marker_color, markersize=6)
        plt.plot(H_span[0, :], tmp[0] * H_span[0, :] + tmp[1], line_color)
        plt.xlim(0, 1)
        plt.xlabel('Hurst exponent H')
        plt.ylabel('Entropy exponent in the r-plane')
        plt.title('RangeEn-B')
        plt.legend(['With correction', 'With correction, p: ' + '%0.2f' % (p_entropy_hurst[3, 0]), 'Withou correction',
                    'Without correction, p: ' + '%0.2f' % (p_entropy_hurst[3, 1])])

        plt.suptitle('Signal type: fBm')


    plt.show()

print('Finished!')

