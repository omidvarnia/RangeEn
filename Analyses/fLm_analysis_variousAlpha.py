# Entropy analysis of fractional Levy motion (fLm) at a fixed Hurst exponent and different alpha values
#
# If the flag 'force' is 1, the code is executed from scratch and may take some time to be finished.
# If the flag 'force' is 0, the code loads pre-saved results in the 'Results' folder.
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import sys, os
from Analyses import sim_data, measures
from os.path import dirname, abspath

############## Set path
main_folder = dirname(dirname(abspath(__file__))) # /Main RangeEn folder
sys.path.insert(0, main_folder)

############## Set parameters
N_single = 1000
m_single = 2
r_span = np.arange(.01, 1, 0.01)  # Span of the tolerance parameter r
r_span = np.reshape(r_span, (1, len(r_span)))
_, N_r = np.shape(r_span)  # Number of r values
force = 0

# Fractional Levy motion
N_alpha = [.8, 1, 1.2, 1.4, 1.6, 1.8] # Levy process parameter: exponential power of the spectral representation of
                                      # the distribution of diff(fLm). alpha of 2 leads to a Gaussian distribution
                                      # for the diff values which is equivalent with fBm.
                                      # See Eq. 3 # From Lui et al, A corrected and generalized successive random
                                      # additions algorithm for simulating Fractional Levy Motion,
                                      # Mathematical Geology, 36 (2004)
H = 0.75    # Hurst exponent of the flm signal
n = 10      # 2^n + 1 will be the length of the flm signal

############## Define the results filename
cwd = os.getcwd() # Current working directory
if(not os.path.exists(cwd + os.sep + 'Results')):
    # Create the Results folder, if needed.
    os.mkdir(cwd + os.sep + 'Results')
output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig4_fLm_N' + str(N_single) + '_variousAlpha.npz'

if (not os.path.isfile(output_filename) or force):
    ############## Initialize ApEn, SampEn and RangeEn
    ApEn_h = np.zeros((len(N_alpha), N_r))
    SampEn_h = np.zeros((len(N_alpha), N_r))
    RangeEn_A_h = np.zeros((len(N_alpha), N_r))
    RangeEn_B_h = np.zeros((len(N_alpha), N_r))

    for n_alpha in range(0,6):

        ############## Simulate fractional Levy motion
        ## Ref: https://github.com/cpgr/flm
        alpha = N_alpha[n_alpha]
        x = sim_data.fLm(alpha, H, n, dim=1, nm=10)
        x = x[0:N_single]

        ############## Calculate ApEn, SampEn and RangeEn
        t00 = time.time()
        for n_r in range(0, N_r):
            ApEn_h[n_alpha, n_r] = measures.ApEn(x, m_single, r_span[0, n_r])

            # Sample Entropy
            SampEn_h[n_alpha, n_r] = measures.SampEn(x, m_single, r_span[0, n_r])

            # RangeEn-A (Modified Approximate Entropy)
            RangeEn_A_h[n_alpha, n_r] = measures.RangeEn_A(x, m_single, r_span[0, n_r])

            # RangeEn-B (Modified Sample Entropy)
            RangeEn_B_h[n_alpha, n_r] = measures.RangeEn_B(x, m_single, r_span[0, n_r])
            print('r = ' + str(r_span[0, n_r]) + ', alpha = ' + str(alpha))

        print('alpha = ' + str(alpha) + 'was finished. Entire elapsed time = ' + str(time.time() - t00))

    ##### Save the entropy values in an external .npz file
    np.savez(output_filename, SampEn_h=SampEn_h, ApEn_h=ApEn_h, RangeEn_B_h=RangeEn_B_h, RangeEn_A_h=RangeEn_A_h)

else:
    ##### Load the existing output .npz file
    out         = np.load(output_filename)
    SampEn_h    = out['SampEn_h']
    ApEn_h      = out['ApEn_h']
    RangeEn_B_h = out['RangeEn_B_h']
    RangeEn_A_h = out['RangeEn_A_h']

############## Plot a sample fBm and its related measures
# plt.figure()
# # set height ratios for sublots
# plt.subplot(3, 1, 1)
# plt.plot(x)
# plt.title('D = ' + str(D))
#
# plt.subplot(3, 2, 3)
# plt.plot(r_span[0,:], ApEn_h)
# ax = plt.gca()
# ax.set_xscale("log", nonposx='clip')
# ax.autoscale(enable=True, axis='x', tight=True)
# plt.title('ApEn')
#
# plt.subplot(3, 2, 4)
# plt.plot(r_span[0,:], SampEn_h)
# ax = plt.gca()
# ax.set_xscale("log", nonposx='clip')
# ax.autoscale(enable=True, axis='x', tight=True)
# plt.title('SampEn')
#
# plt.subplot(3, 2, 5)
# plt.plot(r_span[0,:], RangeEn_A_h)
# ax = plt.gca()
# ax.set_xscale("log", nonposx='clip')
# ax.autoscale(enable=True, axis='x', tight=True)
# plt.title('RangeEn A')
# plt.xlabel('r')
#
# plt.subplot(3, 2, 6)
# plt.plot(r_span[0,:], RangeEn_B_h)
# ax = plt.gca()
# ax.set_xscale("log", nonposx='clip')
# ax.autoscale(enable=True, axis='x', tight=True)
# plt.title('RangeEn B')
# plt.xlabel('r')
#
# plt.show()

############### Plot all Ds
plt.figure()
cmap = plt.get_cmap('jet', len(N_alpha))

### ApEn
plt.subplot(2, 2, 1)
ax = plt.gca()
ax.set_xscale("log", nonposx='clip')
ax.autoscale(enable=True, axis='x', tight=True)
for n_D, n in enumerate(np.linspace(0, 1, len(N_alpha))):
    ax.plot(r_span[0, :], ApEn_h[n_D, :], c=cmap(n_D))
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm)
plt.ylim([0, .8])
plt.ylabel('ApEn'), plt.title('ApEn')

### SampEn
plt.subplot(2, 2, 2)
ax = plt.gca()
ax.set_xscale("log", nonposx='clip')
ax.autoscale(enable=True, axis='x', tight=True)
for n_D, n in enumerate(np.linspace(0, 1, len(N_alpha))):
    ax.plot(r_span[0, :], SampEn_h[n_D, :], c=cmap(n_D))
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm)
plt.ylim([0, 4])
plt.ylabel('SampEn'), plt.title('SampEn')

### RangeEn A
plt.subplot(2, 2, 3)
ax = plt.gca()
ax.set_xscale("log", nonposx='clip')
ax.autoscale(enable=True, axis='x', tight=True)
for n_D, n in enumerate(np.linspace(0, 1, len(N_alpha))):
    ax.plot(r_span[0, :], RangeEn_A_h[n_D, :], c=cmap(n_D))
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm)
plt.ylim([0, .8])
plt.xlabel('r'), plt.ylabel('RangeEn-A'), plt.title('RangeEn-A')

plt.subplot(2, 2, 4)
ax = plt.gca()
ax.set_xscale("log", nonposx='clip')
ax.autoscale(enable=True, axis='x', tight=True)
for n_D, n in enumerate(np.linspace(0, 1, len(N_alpha))):
    ax.plot(r_span[0, :], RangeEn_B_h[n_D, :], c=cmap(n_D))
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm)
plt.ylim([0, .8])
plt.xlabel('r'), plt.ylabel('RangeEn-B'), plt.title('RangeEn-B')

plt.show()

print('Finished!')