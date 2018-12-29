function y = SampEn_slow(x, m, r)
% Slow implementation of sample entropy (SampEn).
%
% Written by: Amir Omidvarnia, PhD
% Email: a.omidvarnia@brain.org.au
% 
% Reference: J. S. Richman and J. R. Moorman, “Physiological time-series analysis using approximate entropy and sample entropy,” 
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
%       y = SampEn_slow(x, 5, 0.2);
N = length(x);
B_m_r = zeros(1,2);
m2 = m;

for ss = 1 : 2
    m2 = m2 + (ss-1);
    B_m_i = zeros(1,(N-m2));
    for i = 1 : (N-m2)
        x_i = x(i:(i+m2-1)); % Template
        d = zeros(1,(N-m2));
        for j = 1 : (N-m2)
            x_j = x(j:(j+m2-1)); % Template
            d(j) = max(abs(x_i - x_j)); % d_chebyshev
        end
        d(i) = [];
        B_m_i(i) = sum(d<=r);
    end
    B_m_r(ss) = mean(B_m_i);
end

y = log(B_m_r(1)) - log(B_m_r(2));

if(isinf(y))
    y = [];
end