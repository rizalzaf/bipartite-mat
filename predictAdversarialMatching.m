function [ Y_pred, Y_prob ] = predictAdversarialMatching( X, theta, mu )
%Predict Adversarial Bipartite Matching
%Maximization
%   X : 4d tensor #feature x #node x #node x #case
%   mu : smoothing parameter 
%   theta : learned parameter #feature 
%   P : 3d tensor #node x #node x #case
%   Y_pred : 0-1 prediction 3d tensor #node x #node x #case
%   Y_prob : probability prediction 3d tensor #node x #node x #case

[~, nn, ~, nc] = size(X);

Y_pred = zeros(nn, nn, nc);
Y_prob = zeros(nn, nn, nc);

% random init P
P_init = zeros(nn, nn, nc);
for i = 1:nc
    P_init(:,:,i) = random_bistochastic(nn);
end

% objective
funObj = @(p)predictionMatchingObj(reshape(p, [nn, nn, nc]), theta, X, mu);

% projection
funProj = @(p)projectPBisthochastic(reshape(p, [nn, nn, nc]));

% run, find P
options = [];
options.maxIter = 500;
options.optTol = 1e-10;
options.projTol = 1e-10;
options.progTol = 1e-15;

p_init = P_init(:);
% p_opt = minConf_PQN(funObj,p_init,funProj,options);
p_opt = minConf_SPG(funObj,p_init,funProj,options);
P_opt = reshape(p_opt, [nn, nn, nc]);

% prob prediction
Y_prob = P_opt;

% prediction
for i = 1:nc    
    P_nl = -log(P_opt(:,:,i));
    P_nl(P_nl == Inf) = intmax;  % avoid infinity
    matching = munkres(P_nl);
        
    P_pred = zeros(nn, nn);
    idr = 1:nn;
    P_pred(sub2ind(size(P_pred), idr, matching)) = 1;
    
    Y_pred(:,:,i) = P_pred;
end

end

function [ vz ] = projectPBisthochastic( P )
%   Q : 3d tensor #node x #node x #cases

[nn, ~, nc] = size(P);

Z = zeros(nn, nn, nc);

for i = 1:nc
    Pi = P(:,:,i);
    Zi = projectBisthochasticADMM(Pi);
    Z(:,:,i) = Zi;
end

vz = Z(:);

end
