# Entropy analysis of epileptic EEG datasets over the range of r values (0 to 1)
#
# This script generates Fig. 9-B to E of the below manuscript:
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

############## Set path
main_folder = dirname(dirname(abspath(__file__))) # /Main RangeEn folder
sys.path.insert(0, main_folder)

EEG_folder = main_folder + os.sep + 'EEG_data' + os.sep + 'datasets'

from Analyses import measures

############## Set parameters
is_plot  = 1      # Plotting flag (if 1, the final result will be plotted).
force    = 0     # BE CAREFULL! If 1, the analysis will be done, even if there is an existing result file.

N_seg   = 100    # Number of EEG segments in each dataset
m_single = 2      # Embeding dimension

r_span   = np.arange(.01, 1, 0.01)               # Span of the tolerance parameter r
r_span   = np.reshape(r_span, (1, len(r_span)))

# This is a flag for parallel processing (suitable for multi-core computers or high performance computing units (HPCUs).
# Note that both processing pipelines will generate the same results, but parallet processing is much faster.
processing_type = 'parallel' # 'parallel' or 'non-parallel'
n_processor     = 32

# EEG dataset labels in Andrzejak et al, 2001:
# label 'Z' associated with dataset 'A': scalp EEG of 5 healthy subjects with eyes open
# label 'O' associated with dataset 'B': scalp EEG of 5 healthy subjects with eyes closed
# label 'N' associated with dataset 'C': intracraninal interictal EEG of 5 epilepsy subjects from the contralateral hippocampal area
# label 'F' associated with dataset 'D': intracraninal interictal EEG of 5 epilepsy subjects from the ipsilateral hippocampal area
# label 'S' associated with dataset 'E': intracraninal ictal EEG of 5 epilepsy subjects from all ictal regions
EEG_database_label = 'S'
if(EEG_database_label=='N'): # Aparently, numpy.loadtxt is case sensitive and doesn't treat xx.txt and xx.TXT the same!
    file_ext = '.TXT'
else:
    file_ext = '.txt'
STD_correction = 'yes' # Correction of the signal amplitude for ApEn and SampEn through dividing the input signal by its STD: 'yes' or 'no'

############## Define the results filename
cwd = os.getcwd()  # Current working directory
if(not os.path.exists(cwd + os.sep + 'Results')):
    # Create the Results folder, if needed.
    os.mkdir(cwd + os.sep + 'Results')

if(STD_correction=='yes'):
    output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig8_EEG_dataset_' + EEG_database_label + '_' + processing_type + '_STDcorrection.npz'
else:
    output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig8_EEG_dataset_' + EEG_database_label + '_' + processing_type + '.npz'

################ FOR PARALLEL PROCESSING: Analysis function for extracting entropy measures from a typical EEG segment
def Entropy_analysis(parallel_inputs):
    t0 = time.time()

    # Unpack the parallel input
    x        = parallel_inputs['x']          # Input EEG segment
    r        = parallel_inputs['r']          # Tolerance value
    emb_dim  = parallel_inputs['emb_dim']    # Embedding dimension

    ### Perform entropy analysis
    ApEn = measures.ApEn(x, emb_dim, r)

    # Sample Entropy
    SampEn = measures.SampEn(x, emb_dim, r)

    # RangeEn-A (Modified Approximate Entropy)
    RangeEn_A = measures.RangeEn_A(x, emb_dim, r)

    # RangeEn-B (Modified Sample Entropy)
    RangeEn_B = measures.RangeEn_B(x, emb_dim, r)

    print('Entropy analysis of the EEG segment was finished! Elapsed time: ' + str(time.time()-t0))

    return ApEn, SampEn, RangeEn_A, RangeEn_B

def entropy_measure_conditioning(ApEn_r, SampEn_r, RangeEn_A_r, RangeEn_B_r, N_r):
    ############## Remove NaN and Info values from the entropy measures
    ApEn_r2_m = []  # Mean of ApEn over permutaions at each tolerance r
    ApEn_r2_std = []  # Std of ApEn over permutaions at each tolerance r
    def_ind1 = []  # Indices of real-valued mean ApEn over permutaions at each tolerance r

    SampEn_r2_m = []  # Mean of SampEn over permutaions at each tolerance r
    SampEn_r2_std = []  # Std of SampEn over permutaions at each tolerance r
    def_ind2 = []  # Indices of real-valued mean SampEn over permutaions at each tolerance r

    RangeEn_A_r2_m = []  # Mean of RangeEn-A over permutaions at each tolerance r
    RangeEn_A_r2_std = []  # Std of RangeEn-A over permutaions at each tolerance r
    def_ind3 = []  # Indices of real-valued mean RangeEn-A over permutaions at each tolerance r

    RangeEn_B_r2_m = []  # Mean of RangeEn-B over permutaions at each tolerance r
    RangeEn_B_r2_std = []  # Std of RangeEn-B over permutaions at each tolerance r
    def_ind4 = []  # Indices of real-valued mean RangeEn-B over permutaions at each tolerance r

    for n_r in range(0, N_r):
        # ApEn
        undef = np.isnan(ApEn_r[:, n_r]) + np.isinf(ApEn_r[:, n_r])
        ind = [i for i, x in enumerate(undef.flatten()) if x == False]
        if (ind):
            ApEn_r2_m.append(np.mean(ApEn_r[ind, n_r]))
            ApEn_r2_std.append(np.std(ApEn_r[ind, n_r]))
            def_ind1.append(n_r)

        # SampEn
        undef = np.isnan(SampEn_r[:, n_r]) + np.isinf(SampEn_r[:, n_r])
        ind = [i for i, x in enumerate(undef.flatten()) if x == False]
        if (ind):
            SampEn_r2_m.append(np.mean(SampEn_r[ind, n_r]))
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

    ApEn_r2_m = np.array(ApEn_r2_m)
    ApEn_r2_std = np.array(ApEn_r2_std)

    SampEn_r2_m = np.array(SampEn_r2_m)
    SampEn_r2_std = np.array(SampEn_r2_std)

    RangeEn_A_r2_m = np.array(RangeEn_A_r2_m)
    RangeEn_A_r2_std = np.array(RangeEn_A_r2_std)

    RangeEn_B_r2_m = np.array(RangeEn_B_r2_m)
    RangeEn_B_r2_std = np.array(RangeEn_B_r2_std)

    out1 = {"1": ApEn_r2_m, "2": ApEn_r2_std, "3": def_ind1}
    out2 = {"1": SampEn_r2_m, "2": SampEn_r2_std, "3": def_ind2}
    out3 = {"1": RangeEn_A_r2_m, "2": RangeEn_A_r2_std, "3": def_ind3}
    out4 = {"1": RangeEn_B_r2_m, "2": RangeEn_B_r2_std, "3": def_ind4}

    return out1, out2, out3, out4

if (processing_type == 'parallel'):
    ### Import the multiprocessing library, if needed.
    import multiprocessing
    pool = multiprocessing.Pool(processes=n_processor)

############### Main script
if(__name__=='__main__'): # If this is the main script (and not a called function within the main script)
    # Perform entropy analysis over the span of r values: See also Fig.1-B of Richman and Moorman 2000
    _, N_r = np.shape(r_span)         # Number of r values

    if (not os.path.isfile(output_filename) or force):

        ApEn_r      = np.zeros((N_seg, N_r))
        SampEn_r    = np.zeros((N_seg, N_r))
        RangeEn_A_r = np.zeros((N_seg, N_r))
        RangeEn_B_r = np.zeros((N_seg, N_r))

        #### Parallel processing
        if (processing_type == 'parallel'):

            t00 = time.time()

            ### Run entropy analysis
            for n_r in range(0, N_r):

                t0 = time.time()
                print('*** Parallel analysis of tolerance ' + str(r_span[0, n_r]) + ' started.')

                # EEG segments are packed for parallel processing
                parallel_inputs = []  # The matrix of all EEG segments in the dataset
                for n_seg in range(0, N_seg):

                    # Load the EEG segment
                    segment_filename = os.path.join(EEG_folder + os.sep + EEG_database_label, EEG_database_label + str("%03d" % (n_seg + 1)) + file_ext)
                    x = np.loadtxt(segment_filename)

                    if (STD_correction == 'yes'):
                        x = x/np.std(x)

                    EEG_inputs = {"x": x, "emb_dim": m_single, 'r': r_span[0, n_r]}

                    parallel_inputs.append(EEG_inputs)

                result = pool.map(Entropy_analysis, parallel_inputs)  # Parallel processing outcome

                # Unpack the parallel processing results
                for n_seg in range(0, N_seg):

                    result_EEG_seg = result[n_seg]
                    ApEn_r[n_seg, n_r] = result_EEG_seg[0]
                    SampEn_r[n_seg, n_r] = result_EEG_seg[1]
                    RangeEn_A_r[n_seg, n_r] = result_EEG_seg[2]
                    RangeEn_B_r[n_seg, n_r] = result_EEG_seg[3]

                ##### Save the entropy values in an external .npz file
                np.savez(output_filename, SampEn_r=SampEn_r, ApEn_r=ApEn_r, RangeEn_B_r=RangeEn_B_r, RangeEn_A_r=RangeEn_A_r)

                print('*** Tolerance ' + str(r_span[0, n_r]) + ', elapsed time = ' + str(time.time() - t0))

            print('Entire elapsed time = ' + str(time.time() - t00))

        else:  # Non-parallel processing

                t00         = time.time()
                for n_r in range(0,N_r):

                    t0 = time.time()
                    print('*** Non-parallel analysis of tolerance ' + str(r_span[0, n_r]) + ' started.')
                    for n_seg in range(0, N_seg):

                        # Load the EEG segment
                        segment_filename = os.path.join(EEG_folder + os.sep + EEG_database_label, EEG_database_label + str("%03d" %(n_seg+1)) + file_ext)
                        x                = np.loadtxt(segment_filename)
                        if (STD_correction == 'yes'):
                            x = x/np.std(x)

                        # Approximate Entropy
                        ApEn_r[n_seg, n_r]      = measures.ApEn(x, m_single, r_span[0, n_r])

                        # Sample Entropy
                        SampEn_r[n_seg, n_r]    = measures.SampEn(x, m_single, r_span[0, n_r])

                        # RangeEn-A (Modified Approximate Entropy)
                        RangeEn_A_r[n_seg, n_r] = measures.RangeEn_A(x, m_single, r_span[0, n_r])

                        # RangeEn-B (Modified Sample Entropy)
                        RangeEn_B_r[n_seg, n_r] = measures.RangeEn_B(x, m_single, r_span[0, n_r])

                        print('Segment ' + str(n_seg+1) + ' was processed.')

                    ##### Save the entropy values in an external .npz file
                    np.savez(output_filename, SampEn_r=SampEn_r, ApEn_r=ApEn_r, RangeEn_B_r=RangeEn_B_r, RangeEn_A_r=RangeEn_A_r)

                    print('*** Tolerance ' + str(r_span[0, n_r]) + ', elapsed time = ' + str(time.time()-t0))

                print('Entire elapsed time = ' + str(time.time() - t00))

    else:
        ##### Load the existing output .npz file
        out         = np.load(output_filename)
        ApEn_r      = out['ApEn_r']
        SampEn_r    = out['SampEn_r']
        RangeEn_A_r = out['RangeEn_A_r']
        RangeEn_B_r = out['RangeEn_B_r']

    ############## Remove NaN and Info values from the entropy measures
    out1, out2, out3, out4         = entropy_measure_conditioning(ApEn_r, SampEn_r, RangeEn_A_r, RangeEn_B_r, N_r)

    ApEn_r2_m   = out1['1']
    ApEn_r2_std = out1['2']
    def_ind1    = out1['3']

    SampEn_r2_m   = out2['1']
    SampEn_r2_std = out2['2']
    def_ind2    = out2['3']

    RangeEn_A_r2_m   = out3['1']
    RangeEn_A_r2_std = out3['2']
    def_ind3    = out3['3']

    RangeEn_B_r2_m   = out4['1']
    RangeEn_B_r2_std = out4['2']
    def_ind4    = out4['3']

    ############## Plot the resulting entropy patterns over tolerance r values
    if (is_plot):

        dataset_labels = ['Z', 'O', 'N', 'F', 'S']
        legend_txt = ['Z: sEEG, healthy, eyes open',
                      'O: sEEG, healthy, eyes closed',
                      'N: iEEG, interictal, contra',
                      'F: iEEG, interictal, ipsi',
                      'S: iEEG, ictal']

        dataset_labels = ['N', 'F', 'S']
        legend_txt = ['N: iEEG, interictal, contra',
                      'F: iEEG, interictal, ipsi',
                      'S: iEEG, ictal']

        N_dataset = len(dataset_labels)

        ######################## Version 2: Combined epileptic and normal EEG figures
        plt.figure()
        #### Fig 6-A: ApEn
        for n_dataset in range(0, N_dataset):
            ##### Load the existing output .npz file
            if (STD_correction == 'yes'):
                output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig8_EEG_dataset_' + dataset_labels[n_dataset] + '_' + processing_type + '_STDcorrection.npz'
            else:
                output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig8_EEG_dataset_' + dataset_labels[n_dataset] + '_' + processing_type + '.npz'

            out = np.load(output_filename)
            ApEn_r = out['ApEn_r']
            SampEn_r = out['SampEn_r']
            RangeEn_A_r = out['RangeEn_A_r']
            RangeEn_B_r = out['RangeEn_B_r']

            ## Entropy measure trimming
            out1, out2, out3, out4 = entropy_measure_conditioning(ApEn_r, SampEn_r, RangeEn_A_r, RangeEn_B_r, N_r)

            ApEn_r2_m = out1['1']
            ApEn_r2_std = out1['2']
            def_ind1 = out1['3']

            SampEn_r2_m = out2['1']
            SampEn_r2_std = out2['2']
            def_ind2 = out2['3']

            RangeEn_A_r2_m = out3['1']
            RangeEn_A_r2_std = out3['2']
            def_ind3 = out3['3']

            RangeEn_B_r2_m = out4['1']
            RangeEn_B_r2_std = out4['2']
            def_ind4 = out4['3']

            # ApEn
            ax = plt.subplot(221)
            ax.set_xscale("log", nonposx='clip')
            plt.xlim([0.01, 1])
            plt.ylim([0, 1.7])
            plt.errorbar(np.array(r_span[0, def_ind1]), ApEn_r2_m, yerr=ApEn_r2_std, fmt='-^', markersize=2)
            plt.title('ApEn')
            plt.xlabel('r')
            plt.legend(legend_txt)

            # SampEn
            ax = plt.subplot(222)
            ax.set_xscale("log", nonposx='clip')
            plt.xlim([0.01, 1])
            plt.ylim([0, 3.5])
            plt.errorbar(np.array(r_span[0, def_ind2]), SampEn_r2_m, yerr=SampEn_r2_std, fmt='-^', markersize=2)
            plt.title('SampEn')
            plt.xlabel('r')
            plt.legend(legend_txt)

            # RangeEn A
            ax = plt.subplot(223)
            ax.set_xscale("log", nonposx='clip')
            plt.xlim([0.01, 1])
            plt.ylim([0, 1.6])
            plt.errorbar(np.array(r_span[0, def_ind3]), RangeEn_A_r2_m, yerr=RangeEn_A_r2_std, fmt='-^', markersize=2)
            plt.title('RangeEn_A')
            plt.xlabel('r')
            plt.legend(legend_txt)

            # RangeEn B
            ax = plt.subplot(224)
            ax.set_xscale("log", nonposx='clip')
            plt.xlim([0.01, 1])
            plt.ylim([0, 2.5])
            plt.errorbar(np.array(r_span[0, def_ind4]), RangeEn_B_r2_m, yerr=RangeEn_B_r2_std, fmt='-^', markersize=2)
            plt.title('RangeEn_B')
            plt.xlabel('r')
            plt.legend(legend_txt)

        plt.suptitle('EEG - STD correction: ' + STD_correction + ', dataset: ' + EEG_database_label)

        ######################## Histograms of all measures
        # N_bins = 20
        # plt.figure()
        # #### Fig 6-A: ApEn
        # for n_dataset in range(0, N_dataset):
        #     ##### Load the existing output .npz file
        #     if (STD_correction == 'yes'):
        #         output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig8_EEG_dataset_' + dataset_labels[n_dataset] + '_' + processing_type + '_STDcorrection.npz'
        #     else:
        #         output_filename = cwd + os.sep + 'Results' + os.sep + 'Fig8_EEG_dataset_' + dataset_labels[n_dataset] + '_' + processing_type + '.npz'
        #     out = np.load(output_filename)
        #     ApEn_r = out['ApEn_r']
        #     SampEn_r = out['SampEn_r']
        #     RangeEn_A_r = out['RangeEn_A_r']
        #     RangeEn_B_r = out['RangeEn_B_r']
        #
        #     ## Entropy measure trimming
        #     out1, out2, out3, out4 = entropy_measure_conditioning(ApEn_r, SampEn_r, RangeEn_A_r, RangeEn_B_r, N_r)
        #
        #     ApEn_r2_m = out1['1']
        #     ApEn_r2_std = out1['2']
        #     def_ind1 = out1['3']
        #
        #     SampEn_r2_m = out2['1']
        #     SampEn_r2_std = out2['2']
        #     def_ind2 = out2['3']
        #
        #     RangeEn_A_r2_m = out3['1']
        #     RangeEn_A_r2_std = out3['2']
        #     def_ind3 = out3['3']
        #
        #     RangeEn_B_r2_m = out4['1']
        #     RangeEn_B_r2_std = out4['2']
        #     def_ind4 = out4['3']
        #
        #     ax = plt.subplot(221)
        #     plt.hist(ApEn_r[15, def_ind1], bins=N_bins, range=(0, 1), alpha=.6)
        #     plt.title('ApEn')
        #     plt.xlabel('r')
        #     plt.legend(legend_txt)
        #
        #     ax = plt.subplot(222)
        #     plt.hist(SampEn_r[8, def_ind2], bins=N_bins, range=(0, 1), alpha=.6)
        #     plt.title('SampEn')
        #     plt.xlabel('r')
        #     plt.legend(legend_txt)
        #
        #     ax = plt.subplot(223)
        #     plt.hist(RangeEn_A_r[20, def_ind3], bins=N_bins, range=(0, 1), alpha=.6)
        #     plt.title('RangeEn_A')
        #     plt.xlabel('r')
        #     plt.legend(legend_txt)
        #
        #     ax = plt.subplot(224)
        #     plt.hist(RangeEn_B_r[20, def_ind4], bins=N_bins, range=(0, 1), alpha=.6)
        #     plt.title('RangeEn_B')
        #     plt.xlabel('r')
        #     plt.legend(legend_txt)
        #
        # plt.suptitle('EEG - STD correction: ' + STD_correction + ', dataset: ' + EEG_database_label)
        plt.show()

    print('Finished!')


