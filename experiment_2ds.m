
% load dataset
data = 'Pedcross2-Sunnyday';

disp(data);
load(strcat('data/', data, '.mat'));

%rng
rng(1);

% mu
mu = 1e-2;

% parpool
% parpool(5);

fprintf('Dataset %s :: Start Training\n', data);

% get the data
[~, ~, nn, nc] = size(X_train);

% cv 
kf = 5;
folds = kFold(nc, 5);

lambdas = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2];
nls = length(lambdas);

best_lambda = 0.0;
best_acc = -1.0;
for ils = 1:nls
    
    lambda = lambdas(ils);
    cv_acc = zeros(size(X_train,4), 1);
    fprintf('Dataset %s -> Training CV, lambda : %f\n', data, lambdas(ils));
    
    for it = 1:kf
        
        fprintf('Fold %d | \n', it);
    
        id_val = folds{it};
        id_tr = [];
        for i = 1:kf
            if i ~= it
                id_tr = [id_tr folds{i}];
            end
        end

        X_tr = X_train(:,:,:,id_tr);
        Y_tr = Y_train(:,:,id_tr);
        nObj_tr = nObj_train(:,id_tr);
        X_val = X_train(:,:,:,id_val);
        Y_val = Y_train(:,:,id_val);
        nObj_val = nObj_train(:,id_val);

        % training
        [theta, ~] = trainAdversarialMatching(X_tr, Y_tr, lambda, mu);

        [v_acc, ~, ~] = testAdv(X_val, Y_val, nObj_val, theta);
        
        cv_acc(id_val) = v_acc;
    end
    
    acc = mean(cv_acc);
    fprintf('Dataset %s -> Training CV i : %d, lambda : %f, acc : %f\n\n', data, ils, lambda, acc);

    if acc > best_acc
        best_lambda = lambda;
        best_acc = acc;
    end
end


fprintf('\nDataset %s -> Best lambda : %f, cv_acc : %f\n', data, best_lambda, best_acc);
fprintf('Train and evaluate using full data\n');

% Evaluate
[theta, Q] = trainAdversarialMatching(X_train, Y_train, best_lambda, mu);

[v_acc, v_precision, v_recall] = testAdv(X_test, Y_test, nObj_test, theta);
avg_acc = mean(v_acc);
std_acc = std(v_acc);

save(strcat('result/Adv-', data, '.mat'), 'avg_acc', 'std_acc', 'best_lambda', 'theta', 'v_acc', 'v_precision', 'v_recall');

fprintf('Dataset %s => Average Test Accuracy : %f\n\n', data, avg_acc);
fprintf('Dataset %s => SD Test Accuracy : %f\n\n', data, std_acc);


%% 
function [ v_acc, v_precision, v_recall ] = testAdv(X_test, Y_test, nObj_test, theta)

% prediction (non probs)
[~, ~, nn, n_test] = size(X_test);
PS = squeeze(sum(X_test .* theta, 1));
Y_pred = zeros(nn, nn, n_test);
for i = 1:n_test
    % only match #object1 + #object2
    no = sum(sum(nObj_test(:,i)));
    no = nn;
    % use hungarian
    idr = 1:no;
    matching = munkres(-PS(1:no,1:no,i));
    
    % get yhat
    ypr = zeros(no,no);
    ypr(sub2ind(size(ypr), idr, matching)) = 1;
    
    % make full matrix
    ypr_full = eye(nn);
    ypr_full(1:no,1:no) = ypr;
    
    Y_pred(:,:,i) = ypr_full;
end

% evaluate
v_acc = zeros(n_test, 1);
for i = 1:n_test
    % consider only #object1 + #object2
    no = sum(sum(nObj_test(:,i)));
    no = nn;
    
    yte = Y_test(1:no,1:no,i);
    ypr = Y_pred(1:no,1:no,i);
    
    l = dot(1 - yte(:), ypr(:));
    v_acc(i) = (no - l) / no;
end

% precision - recall (tracking)
v_precision = zeros(n_test, 1);
v_recall = zeros(n_test, 1);
for i = 1:n_test
    % consider only #object1 + #object2
    no = sum(nObj_test(:,i));
    
    no1 = nObj_test(1,i);
    nm = 0;
    for j = 1:no1
        nm = nm + sum(Y_test(j,:,i) .* Y_pred(j,:,i));
    end
    prec = nm / no1;
    
    no2 = nObj_test(2,i);
    nm = 0;
    for j = 1:no2
        nm = nm + sum(Y_test(:,j,i) .* Y_pred(:,j,i));
    end
    recall = nm / no2;
    
    v_precision(i) = prec;
    v_recall(i) = recall;
end



end