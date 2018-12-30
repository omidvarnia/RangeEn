% This script extracts approximate entropy (ApEn), sample entropy (SampEn),
% range entropy A (RangeEn-A or modified ApEn) and range entropy B
% (RangeEn-B or modified SampEn) from a group of fractional Brownian motion
% signals with different Hurst exponents and at different tolerence
% values (0 <= r <= 1). For more examples of entropy analysis (but in Python),
% please see: https://github.com/omidvarnia/RangeEn.
%
% Written by: Amir Omidvarnia, PhD
% Email: a.omidvarnia@brain.org.au
%
% Cite as: A. Omidvarnia, M. Mesbah, M. Pedersen, and G. Jackson, “Range Entropy: A Bridge between Signal Complexity and Self-Similarity,”
% Entropy, vol. 20, no. 12, p. 962, Dec. 2018.
%
clear all
clc
close all

%% Initialize the parallel processing pipeline
% You can run the code in the non-parallel mode by commenting out the following
% two lines and also, by replacing 'parfor' in line 46 with 'for'.
% delete(gcp('nocreate'))
% parpool(4);

%% Set the input parameters
N = 1000;                                      % Number of time points for simulation of the fractional Brownian motion signals
m = 2;                                         % Embedding dimension of the reconstructed phase space
sflag = 0;                                     % If the tolerance 'r' should be corrected.
H_span = .01 : .01 : .99;                      % A span of Hurst exponent for entropy measures with fixed signal length (N)
r_span =  0 : .01 : 1;                         % A span of tolerance parameter r for entropy measures with fixed signal length (N)

% Output filename
if(sflag)
    output_filename = ['fBm_entropy_analysis_rCorrected.mat'];
else
    output_filename = ['fBm_entropy_analysis_rNotCorrected.mat'];
end

%% Initialize output matrices
t0 = tic;

if(~exist(output_filename,'file'))
    N_H = length(H_span);           % Number of Hurst levels
    N_r = length(r_span);           % Number of tolerance values
    ApEn_h = cell(N_r,N_H);         % Approximate Entropy
    SampEn_h = cell(N_r,N_H);       % Sample Entropy
    RangeEn_A_h = cell(N_r,N_H);    % RangeEn-A
    RangeEn_B_h = cell(N_r,N_H);    % RangeEn-B
    
    for n_h = 1 : N_H
        
        t1 = tic;
        %%% Generate fBm
        H = H_span(n_h);
        try
            x = wfbm(H,N);
        catch
            H = H - .01;
            H_span(n_h) = H;
            x = wfbm(H,N);
        end
        
        %%% Extract entropy measures
        for n_r = 1 : N_r
            
            t2 = tic;
            
            r = r_span(n_r);
            
            %%% Amplitude correction, if needed
            if(sflag)
                sd_sig = std(x);
                r = r * sd_sig;
            end
            
            %%% Approximate Entropy
            ApEn_h{n_r, n_h} = ApEn_fast(x, m, r);
            
            %%% Sample Entropy
            SampEn_h{n_r, n_h} = SampEn_fast(x, m, r);
            
            %%% RangeEn-A
            RangeEn_A_h{n_r, n_h} = RangeEn_A(x, m, r);
            
            %%% RangeEn-B
            RangeEn_B_h{n_r, n_h} = RangeEn_B(x, m, r);
            
            disp(['* r No. ' num2str(n_r) ', elapsed time: ' num2str(toc(t2))])
        end
        
        disp(['***** H No. ' num2str(n_h) ', elapsed time: ' num2str(toc(t1))])
    end
    
    %%% Save the results
    save(output_filename, 'ApEn_h', 'SampEn_h', 'RangeEn_A_h', 'RangeEn_B_h', 'H_span', 'r_span')
    
else
    load(output_filename)
end

disp(['***** The analysis was finished, elapsed time: ' num2str(toc(t0))])

%% Plot --> Colorbar is associated with the Hurst exponents
figure;
subplot(2,2,1); h = plot(r_span, cell2mat(ApEn_h)); axis tight
set(h, {'color'}, num2cell(jet(length(H_span)), 2));
xlabel('Tolerance r'), ylabel('ApEn')

subplot(2,2,2); h = plot(r_span, cell2mat(SampEn_h)); axis tight
set(h, {'color'}, num2cell(jet(length(H_span)), 2));
xlabel('Tolerance r'), ylabel('SampEn'), colormap jet, colorbar

subplot(2,2,3); h = plot(r_span, cell2mat(RangeEn_A_h)); axis tight
set(h, {'color'}, num2cell(jet(length(H_span)), 2));
xlabel('Tolerance r'), ylabel('RangeEn-A')

subplot(2,2,4); h = plot(r_span, cell2mat(RangeEn_B_h)); axis tight
set(h, {'color'}, num2cell(jet(length(H_span)), 2));
xlabel('Tolerance r'), ylabel('RangeEn-B'), colormap jet, colorbar





