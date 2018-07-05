function [ f, g ] = adversarialMatchingObj( Q, X, Y, lambda, mu )
%Objective function for adversarial matching given Q
%Maximization
%   Q : 3d tensor #node x #node x #case
%   X : 4d tensor #feature x #node x #node x #case
%   Y : 3d tensor #node x #node x #case
%   lambda : regularization parameter
%   mu : smoothing parameter 

[~, nn, ~, nc] = size(X);

% get the optimal theta & P given Q
theta = minTheta(Q, X, Y, lambda);
P = minP(Q, mu);

% calculate function
f = nn * nc - dot(P(:), Q(:));         % loss part
%-- not needed
% f = f + (mu/2) * dot(P(:), P(:));      % smoothing penalty P

% broadcast 1st dimension -> sum 1d -> #node x #node x #case
PSI = squeeze(sum(theta .* X, 1)); 
QmY = Q - Y;
f = f + dot(QmY(:), PSI(:));

f = f - (mu/2) * dot(Q(:), Q(:));      % smoothing penalty Q
f = f / nc;

% regularization
f = f + (lambda/2) * dot(theta, theta);

% gradient
Gr = (PSI - P - mu * Q) ./ nc;
g = Gr(:);

end

function [ P ] = minP( Q, mu )
%Find optimal P
%   min_Pi <Pi, -Qi> + mu/2 || Pi ||_F^2
%   equivalent: min_Pi mu/2 || Pi - 1/mu Qi ||_F^2

[nn, ~, nc] = size(Q);

P = zeros(nn, nn, nc);

for i = 1:nc
    Qi = Q(:,:,i);
    
    % hungarian algorithm for the starting point
    matching = munkres(-Qi);
    idr = 1:nn;
    Pi_init = zeros(nn,nn);
    Pi_init(sub2ind(size(Pi_init), idr, matching)) = 1;
    
    % run ADMM
    Pi = projectBisthochasticADMM(Qi ./ mu, Pi_init);

    % set result
    P(:,:,i) = Pi;
end

end
