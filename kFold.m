function [ folds ] = kFold( n, k )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

idx = randperm(n);

folds = cell(k,1);
n_f = round(floor(n/k));
add_f = mod(n, k);
j = 1;
for i=1:k
    if i <= add_f
        folds{i} = idx(j:j+n_f);
        j = j + n_f+1;
    else
        folds{i} = idx(j:j+n_f-1);
        j = j + n_f;
    end
end

end

