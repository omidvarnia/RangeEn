# Histogram of Hurst exponents from epileptic EEG datasets over the range of r values (0 to 1)
#
# This script generates Fig. 9-A of the below manuscript:
#
# If the flag 'force' is 1, the code is executed from scratch and may take some time to be finished.
# If the flag 'force' is 0, the code loads pre-saved results in the 'Results' folder.
# Change 'EEG_database_label' to choose between different EEG datasets.
#
# A. Omidvarnia, M. Mesbah, M. Pedersen, G. Jackson, Range Entropy: a bridge between signal complexity and self-similarity, Entropy, 2018
#
# EEG dataset is available at: http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3
# Ref: Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001) Indications of nonlinear deterministic and finite dimensional
# structures in time series of brain electrical activity: Dependence on recording region and brain state, Phys. Rev. E, 64, 061907
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
import numpy as np
import os, time
import matplotlib.pyplot as plt
import nolds

############## Set path
main_folder = dirname(dirname(abspath(__file__))) # /Main RangeEn folder
sys.path.insert(0, main_folder)

EEG_folder = main_folder + os.sep + 'EEG_data' + os.sep + 'datasets'

############## Set parameters
is_plot  = 1      # Plotting flag (if 1, the final result will be plotted).
force    = 0      # BE CAREFULL! If 1, the analysis will be done, even if there is an existing result file.

N_seg   = 100    # Number of EEG segments in each dataset

# EEG dataset labels in Andrzejak et al, 2001:
# label 'Z' associated with dataset 'A': scalp EEG of 5 healthy subjects with eyes open
# label 'O' associated with dataset 'B': scalp EEG of 5 healthy subjects with eyes closed
# label 'N' associated with dataset 'C': intracraninal interictal EEG of 5 epilepsy subjects from the contralateral hippocampal area
# label 'F' associated with dataset 'D': intracraninal interictal EEG of 5 epilepsy subjects from the ipsilateral hippocampal area
# label 'S' associated with dataset 'E': intracraninal ictal EEG of 5 epilepsy subjects from all ictal regions
EEG_database_label = 'Z'
if(EEG_database_label=='N'): # Aparently, numpy.loadtxt is case sensitive and doesn't treat xx.txt and xx.TXT the same!
    file_ext = '.TXT'
else:
    file_ext = '.txt'

############## Define the results filename
cwd = os.getcwd()  # Current working directory
if(not os.path.exists(cwd + os.sep + 'Results')):
    # Create the Results folder, if needed.
    os.mkdir(cwd + os.sep + 'Results')
output_filename = cwd + os.sep + 'Results' + os.sep + 'EEG_dataset_' + EEG_database_label + 'Hurst_analysis.npz'

############### Main script
# Perform entropy analysis over the span of r values: See also Fig.1-B of Richman and Moorman 2000

if (not os.path.isfile(output_filename) or force):

    EEG_hurst      = np.zeros((1,N_seg))
    t00         = time.time()

    for n_seg in range(0, N_seg):

        # Load the EEG segment
        segment_filename = os.path.join(EEG_folder + os.sep + EEG_database_label, EEG_database_label + str("%03d" %(n_seg+1)) + file_ext)
        x                = np.loadtxt(segment_filename)

        # Estimate the Hurst exponent
        EEG_hurst[0,n_seg]      = nolds.hurst_rs(x)

        print('Segment ' + str(n_seg+1) + ' was processed.')

    ##### Save the hurst exponents in an external .npz file
    np.savez(output_filename, EEG_hurst=EEG_hurst)

    print('Entire elapsed time = ' + str(time.time() - t00))

else:
    ##### Load the existing output .npz file
    out         = np.load(output_filename)
    EEG_hurst      = out['EEG_hurst']


############## Plot the resulting entropy patterns over tolerance r values
if (is_plot):

    dataset_labels = ['Z', 'O', 'N', 'F', 'S']
    legend_txt = ['Z: sEEG, healthy, eyes open',
                  'O: sEEG, healthy, eyes closed',
                  'N: iEEG, interictal, contra',
                  'F: iEEG, interictal, ipsi',
                  'S: iEEG, ictal']
    ######################## Version 1: Separate epileptic and normal EEG figures
    plt.figure()
    N_bins = 50
    #### Fig 9-A: RangeEn-A and epileptic EEG
    for n_dataset in range(2,5):
        ##### Load the existing output .npz file
        output_filename = cwd + os.sep + 'Results' + os.sep + 'EEG_dataset_' + dataset_labels[n_dataset] + 'Hurst_analysis.npz'

        out = np.load(output_filename)

        EEG_hurst = out['EEG_hurst']

        # hist, bin_edges = np.histogram(EEG_hurst[0,:],N_bins)
        # plt.plot(bin_edges[0:-1],hist)

        if(n_dataset==4):
            plt.hist(EEG_hurst[0,:],bins=N_bins,range=(0,1), alpha=.8 )
        else:
            plt.hist(EEG_hurst[0, :], bins=N_bins, range=(0, 1), alpha=.6)
        # plt.xlim([0.01, 1])
        # plt.ylim([0,1])

    plt.title('Hurst exponents - epileptic EEG')
    plt.xlabel('r')
    plt.legend(legend_txt[2:])
    ax = plt.gca()
    ax.autoscale(enable=True, axis='x', tight=True)

    plt.show()

print('Finished!')


