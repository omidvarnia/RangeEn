# Single-scale analysis of ApEn, SampEn, RangeEn_A and RangeEn_B at different signal lengths N
#
# This script generates Fig. 1 of the below manuscript:
#
# A. Omidvarnia, M. Mesbah, M. Pedersen, G. Jackson, Range Entropy: a bridge between signal complexity and self-similarity, Entropy, 2018
#
# Change the input 'sig_type' for generating Fig 1-A, B and C.
# If the flag 'force' is 1, the code is executed from scratch and may take some time to be finished.
# If the flag 'force' is 0, the code loads pre-saved results in the 'Results' folder.
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
import numpy as np
import os, time
from Analyses import sim_data, measures
import matplotlib.pyplot as plt

############## Set path
main_folder = dirname(dirname(abspath(__file__))) # /Main RangeEn folder
sys.path.insert(0, main_folder)

############## Set parameters
is_plot  = 1      # Plotting flag (if 1, the final result will be plotted).
force    = 0      # BE CAREFULL! If 1, the analysis will be done, even if there is an existing result file.

N_single = 1000   # Number of time points in the input signal
m_single = 2      # Embeding dimension

r_single = 0.2    # Tolerance r
N_surr   = 5      # Number of permutations

N_span   = np.arange(50, N_single, 10)           # Span of the signal lengths
N_span   = np.reshape(N_span, (1, len(N_span)))

# Available types of simulated dignals: 'white_noise', 'pink_noise', 'brown_noise'
sig_type = 'pink_noise'
STD_correction = 'yes' # Correction of the signal amplitude for ApEn and SampEn through dividing the input signal by its STD: 'yes' or 'no'

############## Define the results filename
cwd = os.getcwd() # Current working directory
if(not os.path.exists(cwd + os.sep + 'Results')):
    # Create the Results folder, if needed.
    os.mkdir(cwd + os.sep + 'Results')
output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig1_' + sig_type + '_N' + str(N_single) + '.npz'


############## Perform entropy analysis over the span of N values: See also Fig.1-C of Richman and Moorman 2000
_, N_N = np.shape(N_span)         # Number of N values

if (not os.path.isfile(output_filename) or force):

    ApEn_N      = np.zeros((N_surr, N_N))
    SampEn_N    = np.zeros((N_surr, N_N))
    RangeEn_A_N = np.zeros((N_surr, N_N))
    RangeEn_B_N = np.zeros((N_surr, N_N))

    t00         = time.time()
    for n_N in range(0,N_N):

        t0 = time.time()
        for n_surr in range(0, N_surr):

            # Simulate the input signal
            if(sig_type=='white_noise'):
                x = sim_data.white_noise(N_span[0, n_N])
            elif (sig_type == 'pink_noise'):
                x = sim_data.Pink_noise(N_span[0, n_N])
            elif (sig_type == 'brown_noise'):
                x = sim_data.fBm(int(N_span[0, n_N]), 0.5)

            # Approximate Entropy
            ApEn_N[n_surr, n_N]      = measures.ApEn(x, m_single, r_single)

            # Sample Entropy
            SampEn_N[n_surr, n_N]    = measures.SampEn(x, m_single, r_single)

            # RangeEn-A (Modified Approximate Entropy)
            RangeEn_A_N[n_surr, n_N] = measures.RangeEn_A(x, m_single, r_single)

            # RangeEn-B (Modified Sample Entropy)
            RangeEn_B_N[n_surr, n_N] = measures.RangeEn_B(x, m_single, r_single)

        ##### Save the entropy values in an external .npz file
        np.savez(output_filename, SampEn_N=SampEn_N, ApEn_N=ApEn_N, RangeEn_B_N=RangeEn_B_N, RangeEn_A_N=RangeEn_A_N)

        print('Length ' + str(N_span[0, n_N]) + ', elapsed time = ' + str(time.time()-t0))

    print('Entire elapsed time = ' + str(time.time() - t00))

else:
    ##### Load the existing output .npz file
    out         = np.load(output_filename)
    ApEn_N      = out['ApEn_N']
    SampEn_N    = out['SampEn_N']
    RangeEn_A_N = out['RangeEn_A_N']
    RangeEn_B_N = out['RangeEn_B_N']

############## Remove NaN and Info values from the measures
ApEn_N2_m            = []   # Mean of ApEn over permutaions at each signal length N
ApEn_N2_std          = []   # Std of ApEn over permutaions at each signal length N
def_ind1             = []   # Indices of real-valued mean ApEn over permutaions at each signal length N

SampEn_N2_m          = []   # Mean of SampEn over permutaions at each tolerance r
SampEn_N2_std        = []   # Std of SampEn over permutaions at each tolerance r
def_ind2             = []   # Indices of real-valued mean SampEn over permutaions at each signal length N

RangeEn_A_N2_m       = []   # Mean of RangeEn-A over permutaions at each tolerance r
RangeEn_A_N2_std     = []   # Std of RangeEn-A over permutaions at each tolerance r
def_ind3             = []   # Indices of real-valued mean RangeEn-A over permutaions at each signal length N

RangeEn_B_N2_m       = []   # Mean of RangeEn-B over permutaions at each tolerance r
RangeEn_B_N2_std     = []   # Std of RangeEn-B over permutaions at each tolerance r
def_ind4             = []   # Indices of real-valued mean RangeEn-B over permutaions at each signal length N


for n_N in range(0,N_N):
    # ApEn
    undef = np.isnan(ApEn_N[:, n_N]) + np.isinf(ApEn_N[:, n_N])
    ind   = [i for i, x in enumerate(undef.flatten()) if x == False]
    if (ind):
        ApEn_N2_m.append(np.mean(ApEn_N[ind, n_N]))
        ApEn_N2_std.append(np.std(ApEn_N[ind, n_N]))
        def_ind1.append(n_N)

    # SampEn
    undef = np.isnan(SampEn_N[:, n_N]) + np.isinf(SampEn_N[:, n_N])
    ind   = [i for i, x in enumerate(undef.flatten()) if x == False]
    if (ind):
        SampEn_N2_m.append(np.mean(SampEn_N[ind, n_N]))
        SampEn_N2_std.append(np.std(SampEn_N[ind, n_N]))
        def_ind2.append(n_N)

    # RangeEn-A (Modified Approximate Entropy)
    undef = np.isnan(RangeEn_A_N[:, n_N]) + np.isinf(RangeEn_A_N[:, n_N])
    ind   = [i for i, x in enumerate(undef.flatten()) if x == False]
    if (ind):
        RangeEn_A_N2_m.append(np.mean(RangeEn_A_N[ind, n_N]))
        RangeEn_A_N2_std.append(np.std(RangeEn_A_N[ind, n_N]))
        def_ind3.append(n_N)

    # RangeEn-B (Modified Sample Entropy)
    undef = np.isnan(RangeEn_B_N[:, n_N]) + np.isinf(RangeEn_B_N[:, n_N])
    ind   = [i for i, x in enumerate(undef.flatten()) if x == False]
    if (ind):
        RangeEn_B_N2_m.append(np.mean(RangeEn_B_N[ind, n_N]))
        RangeEn_B_N2_std.append(np.std(RangeEn_B_N[ind, n_N]))
        def_ind4.append(n_N)

ApEn_N2_m        = np.array(ApEn_N2_m)
ApEn_N2_std      = np.array(ApEn_N2_std)

SampEn_N2_m      = np.array(SampEn_N2_m)
SampEn_N2_std    = np.array(SampEn_N2_std)

RangeEn_A_N2_m   = np.array(RangeEn_A_N2_m)
RangeEn_A_N2_std = np.array(RangeEn_A_N2_std)

RangeEn_B_N2_m   = np.array(RangeEn_B_N2_m)
RangeEn_B_N2_std = np.array(RangeEn_B_N2_std)

############## Plot the resulting entropy patterns over signal lengths N
if (is_plot):

    # For the caps of errorbar to be visible
    plt.rcParams.update({'errorbar.capsize': 2})

    ### fill_between
    plt.figure()
    plt.xscale('symlog')
    plt.fill_between(N_span[0, def_ind1], ApEn_N2_m - ApEn_N2_std, ApEn_N2_m + ApEn_N2_std, facecolor='g', edgecolor='k', alpha=.5)
    plt.fill_between(N_span[0, def_ind2], SampEn_N2_m - SampEn_N2_std, SampEn_N2_m + SampEn_N2_std, facecolor='r', edgecolor='k', alpha=.5)
    plt.fill_between(N_span[0, def_ind3], RangeEn_A_N2_m - RangeEn_A_N2_std, RangeEn_A_N2_m + RangeEn_A_N2_std, facecolor='c', edgecolor='k', alpha=.5)
    plt.fill_between(N_span[0, def_ind4], RangeEn_B_N2_m - RangeEn_B_N2_std, RangeEn_B_N2_m + RangeEn_B_N2_std, facecolor='b', edgecolor='k', alpha=.5)

    plt.xlim(0,1), plt.xlabel('N'), plt.title(sig_type)
    ax = plt.gca()
    ax.autoscale(enable=True, axis='x', tight=True)
    plt.legend(['ApEn', 'SampEn', 'RangeEn-A (mApEn)', 'RangeEn-B (mSampEn)'])
    plt.draw()

    ### errorbar
    plt.figure()
    plt.xscale('symlog')
    plt.errorbar(np.array(N_span[0, def_ind1]), ApEn_N2_m, yerr=ApEn_N2_std, fmt='o', color='g', markersize=4)
    plt.errorbar(np.array(N_span[0, def_ind2]), SampEn_N2_m, yerr=SampEn_N2_std, fmt='o', color='r', markersize=4)
    plt.errorbar(np.array(N_span[0, def_ind3]), RangeEn_A_N2_m, yerr=RangeEn_A_N2_std, fmt='o', color='c', markersize=4)
    plt.errorbar(np.array(N_span[0, def_ind4]), RangeEn_B_N2_m, yerr=RangeEn_B_N2_std, fmt='o', color='b', markersize=4)

    plt.xlim(0, 1), plt.xlabel('N'), plt.title(sig_type)
    ax = plt.gca()
    ax.autoscale(enable=True, axis='x', tight=True)
    plt.legend(['ApEn', 'SampEn', 'RangeEn-A (mApEn)', 'RangeEn-B (mSampEn)'])
    plt.draw()

    ### errorbar
    fig = plt.figure()
    ax  = plt.subplot(111)
    ax.set_xscale("log", nonposx='clip')
    plt.errorbar(np.array(N_span[0, def_ind1]), ApEn_N2_m, yerr=ApEn_N2_std, fmt='-o', color='g', markersize=6)
    plt.errorbar(np.array(N_span[0, def_ind2]), SampEn_N2_m, yerr=SampEn_N2_std, fmt='-d', color='r', markersize=6)
    plt.errorbar(np.array(N_span[0, def_ind3]), RangeEn_A_N2_m, yerr=RangeEn_A_N2_std, fmt='-^', color='k', markersize=8)
    plt.errorbar(np.array(N_span[0, def_ind4]), RangeEn_B_N2_m, yerr=RangeEn_B_N2_std, fmt='-s', color='b', markersize=6)

    plt.xlim([49,1010]), plt.xlabel('N'), plt.title(sig_type)
    plt.legend(['ApEn', 'SampEn', 'RangeEn-A (mApEn)', 'RangeEn-B (mSampEn)'])

    plt.show()

print('Finished!')

