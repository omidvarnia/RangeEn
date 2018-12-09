# RangeEn
This package implements the ApEn, SampEn, RangeEn-A and RangeEn-B measures as well as the results of: A. Omidvarnia, M. Mesbah, M. Pedersen, G. Jackson, "Range entropy: A bridge between signal complexity and self-similarity", Entropy, 2018 (https://arxiv.org/pdf/1809.06500.pdf). The Python scripts for generating the figures 1 to 6 have been copied in the Analyses folder with relevant naming. All results have also been pre-saved into the Analyses/Results sub-folder. The Analyses scripts first check the Results sub-folder to load the existing results. If there is no results there or if the 'force' flag in the Analyses scripts is on, the results will be generated from scratch (and it might take quite a long time for some analyses).

All entropy measures (SampEn, ApEn, RangeEn-A and RangeEn-B) have been implemented in 'measures.py'. All simulated models and signals are generated through 'sim_data.py'.

EEG datasets used in this package are publicaly available at: http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3. EEG datasets have been copied to EEG_data folder and have the following labels according to Andrzejak et al, 2001:
label 'Z' associated with dataset 'A': scalp EEG of 5 healthy subjects with eyes open
label 'O' associated with dataset 'B': scalp EEG of 5 healthy subjects with eyes closed
label 'N' associated with dataset 'C': intracraninal interictal EEG of 5 epilepsy subjects from the contralateral hippocampal area
label 'F' associated with dataset 'D': intracraninal interictal EEG of 5 epilepsy subjects from the ipsilateral hippocampal area
label 'S' associated with dataset 'E': intracraninal ictal EEG of 5 epilepsy subjects from all ictal regions

Reference: Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001) Indications of nonlinear deterministic and finite dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state, Phys. Rev. E, 64, 061907.

Implementation of the RangeEn functions as well as SampEn and ApEn in this package is based on the 'sampen' function of 'nolds' library: https://github.com/CSchoel/nolds, https://pypi.org/project/nolds/. Hurst exponent extraction of EEG data is also based on a 'nolds' function.

Simulation of fractional Brownian motion (fBm) and fractional Levy motion (fLm) is based on the 'nolds' and 'flm' (https://github.com/cpgr/flm) libraries, respectively. Simulation of colour noise types is based on the 'acoustics' library (https://pypi.org/project/acoustics/).

Dependencies: Numpy, Scipy, Matplotlib, os, time, sys, nolds, flm, acoustics
