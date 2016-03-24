load('train.mat') % TODO remove this when submitting
trX = data;
trY = label;

[n_features, n_train] = size(trX);
% normalize to be zero mean, unit variance
mu = mean(trX, 2);
stdev = std(trX, 0, 2);
trX = bsxfun(@rdivide, bsxfun(@minus, trX, mu), stdev);

% SVM fit
version = 0; % 0: hard linear, 1: soft linear, 2: soft poly
if version == 0 % hard margin linear kernel   
    C = 10^6; % large C means hard margin
    h1 = trY*trY'; % d_i * d_j
    h2 = trX'*trX; % X_i * X_j
    H = h1 * h2; % d_i * d_j * X_i * X_j
elseif version ==1 % soft margin linear kernel
    C = 0.1;
    % TODO
else % soft margin polynomial kernel
    C = 0.1;
    % TODO
end

f = -1 * ones(1, n_train);
Aeq = trY'; % labels
Beq = 0; % alpha_i*d_i = 0
lb = zeros(n_train, 1); % alpha lower bound = 0
ub = C * ones(n_train, 1); % and upper bound = C
alpha0 = [];
% options = optimset('LargeScale','off','MaxIter',1000);
[alpha,fval,exitflag]=quadprog(H,f,[],[],Aeq,Beq,lb,ub,alpha0);

% load test data
load('test_ls.mat');
teX = data;
teY = label;
% normalize test data
teX = bsxfun(@rdivide, bsxfun(@minus, teX, mu), stdev);
