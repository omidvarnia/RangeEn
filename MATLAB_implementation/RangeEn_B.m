function y = RangeEn_B(x, emb_dim, r)
% Range entropy B (modified version of sample entropy)
%
% Written by: Amir Omidvarnia, PhD
% Email: a.omidvarnia@brain.org.au
% 
% Cite as: A. Omidvarnia, M. Mesbah, M. Pedersen, and G. Jackson, “Range Entropy: A Bridge between Signal Complexity and Self-Similarity,” 
% Entropy, vol. 20, no. 12, p. 962, Dec. 2018.
%
% Inputs:
%       x : (a 1-d vector) input signal
%       m : (positive integer value)  Embedding dimension
%       r : (non-negative real value) Tolerance parameter
% Output:
%       y : RangeEn-B (y may become undefined)
%
% Example:
%       x = rand(1,1000); 
%       y = RangeEn_B(x, 5, 0.2);

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
        
        % Calculate the Range distance
        tmp1 = max(abs(tVecsM-repmat(tVecsM(i,:),size(tVecsM,1),1)),[],2);
        tmp2 = min(abs(tVecsM-repmat(tVecsM(i,:),size(tVecsM,1),1)),[],2);
        dsts = (tmp1 - tmp2)./(tmp1 + tmp2); 
           
        dsts(i) = []; % Exclude self-matching
        
        % Compute the sum of conditional probabilities
        B_m_r(ss) = B_m_r(ss) + sum(dsts < r)/(N - m - 1);
    end
end

% Compute log of summed probabilities
y = -log(B_m_r(2) / B_m_r(1));
