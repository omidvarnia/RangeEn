function y = ApEn_fast(x, emb_dim, r)
% Fast implementation of approximate entropy.
% The idea behind this code originates from the Python
% implementation of 'sampen' from NOLDS library
% (https://pypi.org/project/nolds/#description).
%
% Written by: Amir Omidvarnia, PhD
% Email: a.omidvarnia@brain.org.au
% 
% Reference of ApEn: S. M. Pincus, “Approximate entropy as a measure of system complexity.,” Proc. Natl. Acad. Sci., vol. 88, no. 6, pp. 2297–2301, Mar. 1991.
%
% Inputs:
%       x : (a 1-d vector) input signal
%       m : (positive integer value)  Embedding dimension
%       r : (non-negative real value) Tolerance parameter
% Output:
%       y : ApEn (y is always defined)
%
% Example:
%       x = rand(1,1000); 
%       y = ApEn_fast(x, 5, 0.2);

N = length(x); % Signal length
tVecs = zeros(N - emb_dim+1, emb_dim + 1);

for i = 1 : (N - emb_dim)
    tVecs(i, :) = x(i:(i + emb_dim));
end

phi_m_r = zeros(1,2);
ss = 0;
for m = [emb_dim (emb_dim + 1)]
    ss = ss + 1;
    
    % Get the matrix that we need for the current m
    tVecsM = tVecs(1:(N - m + 1), 1:m);
    
    % Calculate distances between each pair of template vectors
    C = [];
    for i = 1 : length(tVecsM)
        
        % Calculate the Chebyshev distance
        dsts = max(abs(tVecsM-repmat(tVecsM(i,:),size(tVecsM,1),1)),[],2); % It considers self-matching.
        
        % Compute the conditional probability
        C = [C sum(dsts < r)/(N - m)];
    end
    % compute sum of log probabilities
    phi_m_r(ss) = sum(log(C))/(N - m);
end

y = phi_m_r(1) - phi_m_r(2);
