
% load dataset
data = 'Pedcross2-Sunnyday';

disp(data);
load(strcat('data/', data, '.mat'));

%rng
rng(1);
    
fprintf('Dataset %s :: Start CV Training SSVM\n', data);

% get the data
[nf, nn, ~, nc] = size(X_train);

% cv 
kf = 5;
folds = kFold(nc, 5);

cs = [1e3, 1e2, 1e1, 1e0, 1e-1];
nls = length(cs);

best_c = 0.0;
best_acc = -1.0;
for ils = 1:nls
    
    cv_acc = zeros(size(X_train,4), 1);
    fprintf('Dataset %s -> Training SSVM CV, C : %f\n', data, cs(ils));
    
    for it = 1:kf
        
        fprintf('Fold %d | ', it);
        
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
        
        % setup data
        patterns = {};
        labels = {};
        for i = 1:length(id_tr)
            patterns{i} = X_tr(:,:,:,i);
            labels{i} = Y_tr(:,:,i);
        end

        % Train SVM struct
        parm = {};
        parm.patterns = patterns ;
        parm.labels = labels ;
        parm.lossFn = @lossCB ;
        parm.constraintFn  = @constraintCB ;
        parm.featureFn = @featureCB ;
        parm.dimension = nf ;
        parm.verbose = 0 ;
        model = svm_struct_learn(strjoin({' -c ', num2str(cs(ils)), ' -o 2 -v 1 '}), parm) ;
        w = model.w ;

        [v_acc, ~, ~] = testSSVM(X_val, Y_val, nObj_val, w);
        
        cv_acc(id_val) = v_acc;
    end

    acc = mean(cv_acc);
    fprintf('\nDataset %s -> Training SSVM CV, C : %f, acc : %f\n\n', data, cs(ils), acc);
    
    if acc > best_acc
        best_c = cs(it);
        best_acc = acc;
    end
end

% Evaluate
fprintf('\nDataset %s : %d -> Best C : %f, cv_acc : %f\n', best_c, best_acc);
fprintf('Train and evaluate using full data\n');

% setup data
patterns = {};
labels = {};
for i = 1:nc
    patterns{i} = X_train(:,:,:,i);
    labels{i} = Y_train(:,:,i);
end

% Train SVM struct
parm.patterns = patterns ;
parm.labels = labels ;
parm.lossFn = @lossCB ;
parm.constraintFn  = @constraintCB ;
parm.featureFn = @featureCB ;
parm.dimension = nf ;
parm.verbose = 0 ;
model = svm_struct_learn(strjoin({' -c ', num2str(best_c), ' -o 2 -v 1 '}), parm) ;
w = model.w ;

[v_acc, v_precision, v_recall] = testSSVM(X_test, Y_test, nObj_test, w);
avg_acc = mean(v_acc);
std_acc = std(v_acc);

save(strcat('result/SSVM-', data, '.mat'), 'avg_acc', 'std_acc', 'best_c', 'w', 'v_acc', 'v_precision', 'v_recall');
fprintf('\nDataset %s => Average Test Accuracy : %f\n', data, avg_acc);
fprintf('Dataset %s => SD Test Accuracy : %f\n', data, std_acc);


%%
function [ v_acc, v_precision, v_recall ] = testSSVM(X_test, Y_test, nObj_test, w)

% Prediction
[~, ~, nn, tnc] = size(X_test);
PS = squeeze(sum(X_test .* w, 1));
Y_pred = zeros(nn, nn, tnc);
for i = 1:tnc
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

% loss
v_acc = zeros(tnc, 1);
for i = 1:tnc
    % consider only #object1 + #object2
    no = sum(sum(nObj_test(:,i)));
    no = nn;
    
    yte = Y_test(1:no,1:no,i);
    ypr = Y_pred(1:no,1:no,i);
    
    l = hammingLoss(yte, ypr);
    v_acc(i) = (no - l) / no;
end

% precision - recall (tracking)
v_precision = zeros(tnc, 1);
v_recall = zeros(tnc, 1);
for i = 1:tnc
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

%% Functions

function loss = hammingLoss(y, ybar)
    loss = dot(1 - y(:), ybar(:));
end

% callback functions
function delta = lossCB(param, y, ybar)
    % hamming loss
    delta = hammingLoss(y, ybar);
    
    if param.verbose
        fprintf('delta = loss(%3d, %3d) = %f\n', y, ybar, delta) ;
    end
end

function psi = featureCB(param, x, y)
    nn = size(y, 1);
    xy = x .* reshape(y, [1, nn, nn]);
    psi = sparse(squeeze(sum(sum(xy, 3),2)));
    if param.verbose
        fprintf('w = psi([%8.3f,%8.3f], %3d) = [%8.3f, %8.3f]\n', ...
                x, y, full(psi(1)), full(psi(2))) ;
    end
end

function yhat = constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
    w = model.w;
    PS = squeeze(sum(x .* w, 1));
    
    PS = PS + (1 - y);
    
    % use hungarian
    nn = size(PS, 1);
    idr = 1:nn;
    matching = munkres(-PS);
    
    % get yhat
    yhat = zeros(nn,nn);
    yhat(sub2ind(size(yhat), idr, matching)) = 1;

    if param.verbose
        fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
                model.w, x, y, yhat) ;
    end

end
