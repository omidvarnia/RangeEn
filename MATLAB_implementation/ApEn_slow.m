function y = ApEn_slow(x, m, r)
% Slow implementation of approximate entropy (ApEn).
%
% Written by: Amir Omidvarnia, PhD
% Email: a.omidvarnia@brain.org.au
% 
% Reference: S. M. Pincus, “Approximate entropy as a measure of system complexity.,” 
% Proc. Natl. Acad. Sci., vol. 88, no. 6, pp. 2297–2301, Mar. 1991.
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
%       y = ApEn_slow(x, 5, 0.2);

N = length(x);
phi_m_r = zeros(1,2);
m2 = m;

for ss = 1 : 2
    m2 = m2 + (ss-1);
    C = zeros(1,(N-m2+1));
    for i = 1 : (N-m2+1)
        x_i = x(i:(i+m2-1)); % Template
        d = zeros(1,(N-m2+1));
        for j = 1 : (N-m2+1)
            x_j = x(j:(j+m2-1)); % Template
            d(j) = max(abs(x_i - x_j)); % d_chebyshev
        end
        C(i) = sum(d<=r)/(N-m2+1);
    end
    phi_m_r(ss) = sum(log(C))/(N-m2+1);
end

y = phi_m_r(1) - phi_m_r(2);