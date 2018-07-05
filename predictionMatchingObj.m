function [ f, g ] = predictionMatchingObj( P, theta, X, mu )
%Objective function for prediction game (given P and known theta)
%Minimization
%   P : 3d tensor #node x #node x #case
%   X : 4d tensor #feature x #node x #node x #case
%   theta : Vectore #feature
%   mu : smoothing parameter 

[~, nn, ~, nc] = size(X);

PSI = squeeze(sum(theta .* X, 1)); 

% get the optimal Q given P
Q = maxQ(P, PSI, mu);

% calculate function
f = nn * nc - dot(P(:), Q(:));              % loss part
%-- not needed
% f = f + dot(Q(:), PSI(:));
% f = f - (mu/2) * dot(Q(:), Q(:));      % smoothing penalty Q

f = f + (mu/2) * dot(P(:), P(:));      % smoothing penalty P
f = f / nc;

% gradient
Gr = (-Q + mu * P) ./ nc;
g = Gr(:);

end


function [ Q ] = maxQ( P, PSI, mu )
%Find optimal Q
%   max_Q <-P, Q> + <PSI, Q> - mu/2 || Q ||_F^2
%   equivalent: min_Q <Q, P - PSI> + mu/2 || Q ||_F^2
%   equivalent: min_Q mu/2 || Q - 1/mu (PSI - P) ||_F^2

[nn, ~, nc] = size(P);

for i = 1:nc
    Pi = P(:,:,i);
    PSIi = PSI(:,:,i);

    % hungarian algorithm for the starting point
    matching = munkres(Pi - PSIi);
    idr = 1:nn;
    Qi_init = zeros(nn,nn);
    Qi_init(sub2ind(size(Qi_init), idr, matching)) = 1;

    % run ADMM
    Qi = projectBisthochasticADMM((PSIi - Pi) ./ mu, Qi_init);
    
    % set result
    Q(:,:,i) = Qi;
end

end
