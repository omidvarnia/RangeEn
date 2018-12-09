# Single-scale analysis of ApEn, SampEn, RangeEn_A and RangeEn_B at different tolerance values r
#
# This script generates Fig. 2 of the below manuscript:
#
# A. Omidvarnia, M. Mesbah, M. Pedersen, G. Jackson, Range Entropy: a bridge between signal complexity and self-similarity, Entropy, 2018
#
# Change the input 'sig_type' for generating Fig 2-A, B and C.
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

r_span   = np.arange(.01, 1, 0.01)               # Span of the tolerance parameter r
r_span   = np.reshape(r_span, (1, len(r_span)))

# Available types of simulated dignals: 'white_noise', 'pink_noise', 'brown_noise', 'logistic_map', 'henon_map', 'roessler_osc'
sig_type = 'white_noise'

# N_surr: number of permutations. N_surr will be set to 1 for deterministic systems ('logistic_map', 'henon_map', 'roessler_osc').
if(sig_type=='white_noise'):
    N_surr = 100
    output_file_label = 'Fig3'
elif (sig_type == 'pink_noise'):
    N_surr = 100
    output_file_label = 'Fig3'
elif (sig_type == 'brown_noise'):
    N_surr = 100
    output_file_label = 'Fig3'
elif (sig_type == 'logistic_map'):
    N_surr = 1 # The signal is deterministic
    output_file_label = 'Fig5'
elif (sig_type == 'henon_map'):
    N_surr = 1 # The signal is deterministic
    output_file_label = 'Fig5'
elif (sig_type == 'roessler_osc'):
    N_surr = 1 # The signal is deterministic
    output_file_label = 'Fig5'

############## Define the results filename
cwd = os.getcwd()  # Current working directory
if(not os.path.exists(cwd + os.sep + 'Results')):
    # Create the Results folder, if needed.
    os.mkdir(cwd + os.sep + 'Results')
output_filename = cwd + os.sep + 'Results' + os.sep + output_file_label + '_' + sig_type + '_N' + str(N_single) + '.npz'

############## Perform entropy analysis over the span of r values: See also Fig.1-B of Richman and Moorman 2000
_, N_r = np.shape(r_span)         # Number of r values

if (not os.path.isfile(output_filename) or force):

    ApEn_r      = np.zeros((N_surr, N_r))
    SampEn_r    = np.zeros((N_surr, N_r))
    RangeEn_A_r = np.zeros((N_surr, N_r))
    RangeEn_B_r = np.zeros((N_surr, N_r))

    t00         = time.time()
    for n_r in range(0,N_r):

        t0 = time.time()
        for n_surr in range(0, N_surr):

            # Simulate the input signal
            if(sig_type=='uniform_noise'):
                x = sim_data.Uniform_noise(N_single)
            elif(sig_type=='white_noise'):
                x = sim_data.white_noise(N_single)
            elif (sig_type == 'pink_noise'):
                x = sim_data.Pink_noise(N_single)
            elif (sig_type == 'brown_noise'):
                x = sim_data.fBm(int(N_single), 0.5)
            elif (sig_type == 'fBm025'):
                x = sim_data.fBm(int(N_single), 0.25)
            elif (sig_type == 'MIX'):
                x = sim_data.MIX(int(N_single), 0, 50)
            elif (sig_type == 'logistic_map'):
                x = np.array(sim_data.Logistic_map(N_single))
            elif (sig_type == 'henon_map'):
                x, y = sim_data.Henon_map(N_single)
            elif (sig_type == 'roessler_osc'):
                x, y, z = sim_data.Roessler_osc(N_single, t1=0, t2=50)

            # Approximate Entropy
            ApEn_r[n_surr, n_r]      = measures.ApEn(x, m_single, r_span[0, n_r])

            # Sample Entropy
            SampEn_r[n_surr, n_r]    = measures.SampEn(x, m_single, r_span[0, n_r])

            # RangeEn-A (Modified Approximate Entropy)
            RangeEn_A_r[n_surr, n_r] = measures.RangeEn_A(x, m_single, r_span[0, n_r])

            # RangeEn-B (Modified Sample Entropy)
            RangeEn_B_r[n_surr, n_r] = measures.RangeEn_B(x, m_single, r_span[0, n_r])

        ##### Save the entropy values in an external .npz file
        np.savez(output_filename, SampEn_r=SampEn_r, ApEn_r=ApEn_r, RangeEn_B_r=RangeEn_B_r, RangeEn_A_r=RangeEn_A_r)

        print('Tolerance ' + str(r_span[0, n_r]) + ', elapsed time = ' + str(time.time()-t0))

    print('Entire elapsed time = ' + str(time.time() - t00))

else:
    ##### Load the existing output .npz file
    out         = np.load(output_filename)
    ApEn_r      = out['ApEn_r']
    SampEn_r    = out['SampEn_r']
    RangeEn_A_r = out['RangeEn_A_r']
    RangeEn_B_r = out['RangeEn_B_r']

############## Remove NaN and Info values from the entropy measures
ApEn_r2_m            = []   # Mean of ApEn over permutaions at each tolerance r
ApEn_r2_std          = []   # Std of ApEn over permutaions at each tolerance r
def_ind1             = []   # Indices of real-valued mean ApEn over permutaions at each tolerance r

SampEn_r2_m          = []   # Mean of SampEn over permutaions at each tolerance r
SampEn_r2_std        = []   # Std of SampEn over permutaions at each tolerance r
def_ind2             = []   # Indices of real-valued mean SampEn over permutaions at each tolerance r

RangeEn_A_r2_m       = []   # Mean of RangeEn-A over permutaions at each tolerance r
RangeEn_A_r2_std     = []   # Std of RangeEn-A over permutaions at each tolerance r
def_ind3             = []   # Indices of real-valued mean RangeEn-A over permutaions at each tolerance r

RangeEn_B_r2_m       = []   # Mean of RangeEn-B over permutaions at each tolerance r
RangeEn_B_r2_std     = []   # Std of RangeEn-B over permutaions at each tolerance r
def_ind4             = []   # Indices of real-valued mean RangeEn-B over permutaions at each tolerance r

for n_r in range(0,N_r):
    # ApEn
    undef = np.isnan(ApEn_r[:, n_r]) + np.isinf(ApEn_r[:, n_r])
    ind   = [i for i, x in enumerate(undef.flatten()) if x == False]
    if (ind):
        ApEn_r2_m.append(np.mean(ApEn_r[ind, n_r]))
        ApEn_r2_std.append(np.std(ApEn_r[ind, n_r]))
        def_ind1.append(n_r)

    # SampEn
    undef = np.isnan(SampEn_r[:, n_r]) + np.isinf(SampEn_r[:, n_r])
    ind   = [i for i,x in enumerate(undef.flatten()) if x==False]
    if(ind):
        SampEn_r2_m.append(np.mean(SampEn_r[ind ,n_r]))
        SampEn_r2_std.append(np.std(SampEn_r[ind, n_r]))
        def_ind2.append(n_r)

    # RangeEn_A
    undef = np.isnan(RangeEn_A_r[:, n_r]) + np.isinf(RangeEn_A_r[:, n_r])
    ind   = [i for i, x in enumerate(undef.flatten()) if x == False]
    if (ind):
        RangeEn_A_r2_m.append(np.mean(RangeEn_A_r[ind, n_r]))
        RangeEn_A_r2_std.append(np.std(RangeEn_A_r[ind, n_r]))
        def_ind3.append(n_r)

    # RangeEn_B
    undef = np.isnan(RangeEn_B_r[:, n_r]) + np.isinf(RangeEn_B_r[:, n_r])
    ind   = [i for i, x in enumerate(undef.flatten()) if x == False]
    if (ind):
        RangeEn_B_r2_m.append(np.mean(RangeEn_B_r[ind, n_r]))
        RangeEn_B_r2_std.append(np.std(RangeEn_B_r[ind, n_r]))
        def_ind4.append(n_r)

ApEn_r2_m        = np.array(ApEn_r2_m)
ApEn_r2_std      = np.array(ApEn_r2_std)

SampEn_r2_m      = np.array(SampEn_r2_m)
SampEn_r2_std    = np.array(SampEn_r2_std)

RangeEn_A_r2_m   = np.array(RangeEn_A_r2_m)
RangeEn_A_r2_std = np.array(RangeEn_A_r2_std)

RangeEn_B_r2_m   = np.array(RangeEn_B_r2_m)
RangeEn_B_r2_std = np.array(RangeEn_B_r2_std)

############## Plot the resulting entropy patterns over tolerance r values
if (is_plot):

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_xscale("log", nonposx='clip')
    plt.errorbar(np.array(r_span[0, def_ind1]), ApEn_r2_m, yerr=ApEn_r2_std, fmt='-o', color='g', markersize=6)
    plt.errorbar(np.array(r_span[0,def_ind2]), SampEn_r2_m, yerr=SampEn_r2_std, fmt='-d', color='r', markersize=6)
    plt.errorbar(np.array(r_span[0, def_ind3]), RangeEn_A_r2_m, yerr=RangeEn_A_r2_std, fmt='-^', color='k', markersize=8)
    plt.errorbar(np.array(r_span[0, def_ind4]), RangeEn_B_r2_m, yerr=RangeEn_B_r2_std, fmt='-s', color='b', markersize=6)

    plt.xlim([.005, 2])
    ax = plt.gca()
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)
    plt.xlabel('r'), plt.title(sig_type)

    plt.legend(['ApEn', 'SampEn', 'RangeEn-A (mApEn)', 'RangeEn-B (mSampEn)'])
    plt.show()

print('Finished!')

