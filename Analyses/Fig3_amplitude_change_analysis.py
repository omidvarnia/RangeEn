# Single-scale analysis of ApEn, SampEn, RangeEn_A and RangeEn_B at three signal amplitudes (x_1(t), x_2(t) and x_3(t) )
#
# This script generates Fig. 3 of the below manuscript:
#
# A. Omidvarnia, M. Mesbah, M. Pedersen, G. Jackson, Range Entropy: a bridge between signal complexity and self-similarity, Entropy, 2018
#
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

############# Set parameters
is_plot  = 1      # Plotting flag (if 1, the final result will be plotted).
force    = 0      # BE CAREFULL! If 1, the analysis will be done, even if there is an existing result file.

N_single = 1000   # Number of time points in the input signal
m_single = 2      # Embeding dimension

r_span   = np.arange(.01, 1, 0.01)               # Span of the tolerance parameter r
r_span   = np.reshape(r_span, (1, len(r_span)))

N_surr   = 5      # Number of permutations

# Available types of simulated dignals: 'white_noise'
sig_type = 'white_noise'

############## Define the results filename
cwd = os.getcwd() # Current working directory
if(not os.path.exists(cwd + os.sep + 'Results')):
    # Create the Results folder, if needed.
    os.mkdir(cwd + os.sep + 'Results')
output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig2_' + sig_type + '_N' + str(N_single) + '.npz'

############## Perform entropy analysis for two signal amplitudes
_, N_r = np.shape(r_span)   # Number of r values

if (not os.path.isfile(output_filename) or force):
    #### Simulate the input signals x1(t), x2(t) and x3(t)
    if (sig_type == 'uniform_noise'):
        x1 = sim_data.Uniform_noise(N_single)
    elif (sig_type == 'white_noise'):
        x1 = sim_data.white_noise(N_single)
    elif (sig_type == 'pink_noise'):
        x1 = sim_data.Pink_noise(N_single)
    elif (sig_type == 'brown_noise'):
        x1 = sim_data.fBm(int(N_single), 0.5)
    elif (sig_type == 'fBm025'):
        x1 = sim_data.fBm(int(N_single), 0.25)
    elif (sig_type == 'MIX'):
        x1 = sim_data.MIX(int(N_single), 0, 50)
    elif (sig_type == 'logistic_map'):
        x1 = np.array(sim_data.Logistic_map(N_single))
    elif (sig_type == 'henon_map'):
        x1, y = sim_data.Henon_map(N_single)
    elif (sig_type == 'roessler_osc'):
        x1, y, z = sim_data.Roessler_osc(N_single, t1=0, t2=50)
    elif (sig_type == 'chirp'):
        x1 = sim_data.chirp(N_single)

    x2 = 5 * x1

    x3 = (sim_data.Uniform_noise(int(N_single / 5)), 3 * sim_data.Uniform_noise(int(N_single / 5)), 10 * sim_data.Uniform_noise(int(N_single / 5)),
          4 * sim_data.Uniform_noise(int(N_single / 5)), sim_data.Uniform_noise(int(N_single / 5)))
    x3 = np.concatenate(x3)

    #### Perform entropy analysis
    ApEn_r      = np.zeros((3, N_r))
    SampEn_r    = np.zeros((3, N_r))
    RangeEn_A_r = np.zeros((3, N_r))
    RangeEn_B_r = np.zeros((3, N_r))

    for n_r in range(0,N_r):

        t0 = time.time()

        # Approximate Entropy
        ApEn_r[0,n_r]       = measures.ApEn(x1, m_single, r_span[0, n_r])
        ApEn_r[1,n_r]       = measures.ApEn(x2, m_single, r_span[0, n_r])
        ApEn_r[2, n_r]      = measures.ApEn(x3 / np.std(x3), m_single, r_span[0, n_r])

        # Sample Entropy
        SampEn_r[0, n_r]    = measures.SampEn(x1, m_single, r_span[0, n_r])
        SampEn_r[1, n_r]    = measures.SampEn(x2, m_single, r_span[0, n_r])
        SampEn_r[2, n_r]    = measures.SampEn(x3 / np.std(x3), m_single, r_span[0, n_r])

        # RangeEn-A (Modified Approximate Entropy)
        RangeEn_A_r[0, n_r] = measures.RangeEn_A(x1, m_single, r_span[0, n_r])
        RangeEn_A_r[1, n_r] = measures.RangeEn_A(x2, m_single, r_span[0, n_r])
        RangeEn_A_r[2, n_r] = measures.RangeEn_A(x3, m_single, r_span[0, n_r])

        # RangeEn-B (Modified Sample Entropy)
        RangeEn_B_r[0, n_r] = measures.RangeEn_B(x1, m_single, r_span[0, n_r])
        RangeEn_B_r[1, n_r] = measures.RangeEn_B(x2, m_single, r_span[0, n_r])
        RangeEn_B_r[2, n_r] = measures.RangeEn_B(x3, m_single, r_span[0, n_r])

        ##### Save the entropy values in an external .npz file
        np.savez(output_filename, SampEn_r=SampEn_r, ApEn_r=ApEn_r, RangeEn_B_r=RangeEn_B_r, RangeEn_A_r=RangeEn_A_r)

        print('r = ' + str(r_span[0,n_r]) + ', elapsed time = ' + str(time.time()-t0))

else:
    ##### Load the existing output .npz file
    out         = np.load(output_filename)
    SampEn_r    = out['SampEn_r']
    ApEn_r      = out['ApEn_r']
    RangeEn_B_r = out['RangeEn_B_r']
    RangeEn_A_r = out['RangeEn_A_r']


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
    ind = [i for i, x in enumerate(undef.flatten()) if x == False]
    if (ind):
        ApEn_r2_m.append(np.mean(ApEn_r[ind, n_r]))
        ApEn_r2_std.append(np.std(ApEn_r[ind, n_r]))
        def_ind1.append(n_r)

    # SampEn
    undef = np.isnan(SampEn_r[:, n_r]) + np.isinf(SampEn_r[:, n_r])
    ind = [i for i,x in enumerate(undef.flatten()) if x==False]
    if(ind):
        SampEn_r2_m.append(np.mean(SampEn_r[ind ,n_r]))
        SampEn_r2_std.append(np.std(SampEn_r[ind, n_r]))
        def_ind2.append(n_r)

    # RangeEn_A
    undef = np.isnan(RangeEn_A_r[:, n_r]) + np.isinf(RangeEn_A_r[:, n_r])
    ind = [i for i, x in enumerate(undef.flatten()) if x == False]
    if (ind):
        RangeEn_A_r2_m.append(np.mean(RangeEn_A_r[ind, n_r]))
        RangeEn_A_r2_std.append(np.std(RangeEn_A_r[ind, n_r]))
        def_ind3.append(n_r)

    # RangeEn_B
    undef = np.isnan(RangeEn_B_r[:, n_r]) + np.isinf(RangeEn_B_r[:, n_r])
    ind = [i for i, x in enumerate(undef.flatten()) if x == False]
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

############## Plot the resulting entropy patterns for two amplitudes over r values
if (is_plot):
    #################### Figure 3: (x1(t) and x2(t) and x3(t))
    plt.figure()
    ax = plt.subplot(2, 2, 1)
    ax.set_xscale("log", nonposx='clip'), plt.title(sig_type + ', ApEn')
    plt.plot(r_span[0, def_ind1], ApEn_r[0, def_ind1], 'k^--', linewidth=2, markersize=6)
    plt.plot(r_span[0, def_ind1], ApEn_r[1, def_ind1], 'ro--', linewidth=2, markersize=6)
    plt.plot(r_span[0, def_ind1], ApEn_r[2, def_ind1], 'go--', linewidth=2, markersize=6)
    plt.xlabel('r')  # , plt.legend(['x(t)', '5x(t)'])
    plt.ylim([0,5])
    plt.xlim([0.01,1])
    ax.autoscale(enable=True, axis='x', tight=True)

    ax = plt.subplot(2, 2, 2)
    ax.set_xscale("log", nonposx='clip'), plt.title(sig_type + ', SampEn')
    plt.plot(r_span[0,def_ind2], SampEn_r[0, def_ind2], 'k^--', linewidth=2, markersize=6)
    plt.plot(r_span[0,def_ind2], SampEn_r[1, def_ind2], 'ro--', linewidth=2, markersize=6)
    plt.plot(r_span[0, def_ind2], SampEn_r[2, def_ind2], 'go--', linewidth=2, markersize=6)
    plt.xlabel('r'), plt.legend(['x(t)', 'x_1(t)','x_2(t)'])
    plt.ylim([0, 5])
    plt.xlim([0.01, 1])
    ax.autoscale(enable=True, axis='x', tight=True)

    ax = plt.subplot(2, 2, 3)
    ax.set_xscale("log", nonposx='clip'), plt.title(sig_type + ', RangeEn-a (RangeEn_A)')
    plt.plot(r_span[0, def_ind3], RangeEn_A_r[0, def_ind3], 'k^--', linewidth=2, markersize=6)
    plt.plot(r_span[0, def_ind3], RangeEn_A_r[1, def_ind3], 'ro--', linewidth=2, markersize=6)
    plt.plot(r_span[0, def_ind3], RangeEn_A_r[2, def_ind3], 'go--', linewidth=2, markersize=6)
    plt.xlabel('r')  # , plt.legend(['x(t)', '5x(t)'])
    plt.ylim([0, 5])
    plt.xlim([0.01, 1])
    # ax.autoscale(enable=True, axis='x', tight=True)

    ax = plt.subplot(2, 2, 4)
    ax.set_xscale("log", nonposx='clip'), plt.title(sig_type + ', RangeEn-b (RangeEn_B)')
    plt.plot(r_span[0, def_ind4], RangeEn_B_r[0, def_ind4], 'k^--', linewidth=2, markersize=6)
    plt.plot(r_span[0, def_ind4], RangeEn_B_r[1, def_ind4], 'ro--', linewidth=2, markersize=6)
    plt.plot(r_span[0, def_ind4], RangeEn_B_r[2, def_ind4], 'go--', linewidth=2, markersize=6)
    plt.xlabel('r')#, plt.legend(['x(t)', '5x(t)'])
    plt.ylim([0, 5])
    plt.xlim([0.01, 1])
    ax.autoscale(enable=True, axis='x', tight=True)

    plt.show()

print('Finished!')

