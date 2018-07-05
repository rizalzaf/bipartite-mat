function [ theta ] = minTheta( Q, X, Y, lambda )
%Find optimal theta given Q
%   closed form solution 

[~, nn, ~, nc] = size(X);

QmY = Q - Y;
XtQmY = X .* reshape(QmY, [1, nn, nn, nc]);
XtQmY_sum = sum(sum(sum(XtQmY, 4), 3), 2);

theta = -XtQmY_sum ./ (lambda * nc);

end
