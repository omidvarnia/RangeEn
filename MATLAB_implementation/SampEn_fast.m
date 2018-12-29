function y = SampEn_fast(x, emb_dim, r)
% Fast implementation of sample entropy.
% The idea behind this code originates from the Python
% implementation of 'sampen' from NOLDS library
% (https://pypi.org/project/nolds/#description).
%
% Written by: Amir Omidvarnia, PhD
% Email: a.omidvarnia@brain.org.au
% 
% Reference of SampEn: J. S. Richman and J. R. Moorman, “Physiological time-series analysis using approximate entropy and sample entropy,” 
% Am. J. Physiol. Heart Circ. Physiol., vol. 278, no. 6, pp. H2039-2049, Jun. 2000.
%
% Inputs:
%       x : (a 1-d vector) input signal
%       m : (positive integer value)  Embedding dimension
%       r : (non-negative real value) Tolerance parameter
% Output:
%       y : SampEn (y may become undefined)
%
% Example:
%       x = rand(1,1000); 
%       y = SampEn_fast(x, 5, 0.2);

N = length(x); % Signal length
tVecs = zeros(N - emb_dim+1, emb_dim + 1);

for i = 1 : (N - emb_dim)
    tVecs(i, :) = x(i:(i + emb_dim));
end

B_m_r = zeros(1,2);
ss = 0;
for m = [emb_dim (emb_dim + 1)]
    ss = ss + 1;
    
    % Get the matrix that we need for the current m
    tVecsM = tVecs(1:(N - m + 1), 1:m);
    
    % Calculate distances between each pair of template vectors
    for i = 1 : length(tVecsM)
        
        % Calculate the Chebyshev distance
        dsts = max(abs(tVecsM-repmat(tVecsM(i,:),size(tVecsM,1),1)),[],2);
        dsts(i) = []; % Exclude self-matching
        
        % Compute the sum of conditional probabilities
        B_m_r(ss) = B_m_r(ss) + sum(dsts < r)/(N - m - 1);
    end
end

% Compute log of summed probabilities
y = -log(B_m_r(2) / B_m_r(1));
