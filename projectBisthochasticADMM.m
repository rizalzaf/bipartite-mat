function [ Z ] = projectBisthochasticADMM( X, Z_init )
%Project to Bistochastic matrix
%   Using ADMM 
%   X : Input Matrix 
%   Z_init : initialize solution

n = size(X, 1);

% initialize
if nargin > 1
    Z = Z_init;
    Y = Z_init;
else
    Z = rand(n,n);
    Y = rand(n,n);
end
W = zeros(n,n);

% history
Y_old = Y;

% penalty parameters
par_increase = 2.0;
par_decrease = 2.0;
par_mu = 10.0;

% tolerance
max_iter = 100; 
tol_abs = 1e-4;
tol_rel = 1e-2;

% step size init
rho = 1.0;

iter = 1;
while true
    A = (X + rho * (Y - W)) ./ (1.0 + rho);
    % update Z -> column wise projection 
    for i = 1:n
        Z(:, i) = projectSimplex(A(:,i));
    end
    
    B = (X + rho * (Z + W)) ./ (1.0 + rho);
    % update U -> row wise projection 
    for i = 1:n
        Y(i, :) = projectSimplex(B(i,:));
    end

    W = W + Z - Y;

    % stopping criteria
    norm_r = norm(Z - Y, 'fro');
    norm_s = norm(rho * (Y - Y_old), 'fro');
    eps_primal = n * tol_abs + tol_rel * max(norm(Z, 'fro'), norm(Y, 'fro'));   % sqrt(n*n) = n
    eps_dual = n * tol_abs + tol_rel * norm(rho * W, 'fro');
    
    % update rho
    if norm_r > par_mu * norm_s
        rho = par_increase * rho;
    elseif norm_s > par_mu * norm_r
        rho = rho / par_decrease;
    end

    if iter >= max_iter
        break
    elseif norm_r < eps_primal && norm_s < eps_dual
        break
    end
    
    Y_old = Y;
    iter = iter + 1;
end

%fprintf('# iter : %d\n', iter);
end

