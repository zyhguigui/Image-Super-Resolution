function [S] = L1QP_FeatureSign_Set(X, B, Sigma, beta, gamma)

[dFea, nSmp] = size(X);
nBases = size(B, 2);

% sparse codes of the features
% S = sparse(nBases, nSmp); % original code
S = zeros(nBases, nSmp);    % modified 2017/01/01

A = B'*B + 2*beta*Sigma;

for ii = 1:nSmp
    b = -B'*X(:, ii);
%     [net] = L1QP_FeatureSign(gamma, A, b);
    S(:, ii) = L1QP_FeatureSign_yang(gamma, A, b);
end