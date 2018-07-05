function [ theta, Q ] = trainAdversarialMatching( X, Y, lambda, mu )
%Train Adversarial Bipartite Matching
%Maximization
%   X : 4d tensor #feature x #node x #node x #case
%   Y : 3d tensor #node x #node x #case
%   lambda : regularization parameter
%   mu : smoothing parameter 
%   theta : learned parameter #feature 
%   Q : 3d tensor #node x #node x #case

[~, nn, ~, nc] = size(X);

% random init Q
Q_init = zeros(nn, nn, nc);
% for i = 1:nc
%     Q_init(:,:,i) = random_bistochastic(nn);
% end
% uniform
for i = 1:nc
    Q_init(:,:,i) = ones(nn, nn) ./ nn^2;
end

% objective
funObj = @(q)adversarialMinQObj(reshape(q, [nn, nn, nc]), X, Y, lambda, mu);

% projection
funProj = @(q)projectQBisthochastic(reshape(q, [nn, nn, nc]));

% run, find Q
options = [];
options.maxIter = 500;
options.optTol = 1e-10;
options.projTol = 1e-10;
options.progTol = 1e-15;

q_init = Q_init(:);
% q_opt = minConf_PQN(funObj,q_init,funProj,options);
q_opt = minConf_SPG(funObj,q_init,funProj,options);
Q_opt = reshape(q_opt, [nn, nn, nc]);

% find theta
theta = minTheta(Q_opt, X, Y, lambda);
Q = Q_opt;

end

function [ f, g ] = adversarialMinQObj( Q, X, Y, lambda, mu )
%   Flip from max to min

[fm, gm] = adversarialMatchingObj( Q, X, Y, lambda, mu );

f = -fm;
g = -gm;

end

function [ vz ] = projectQBisthochastic( Q )
%   Q : 3d tensor #node x #node x #cases

[nn, ~, nc] = size(Q);

Z = zeros(nn, nn, nc);

for i = 1:nc
    Qi = Q(:,:,i);
    Zi = projectBisthochasticADMM(Qi);
    Z(:,:,i) = Zi;
end

vz = Z(:);

end
