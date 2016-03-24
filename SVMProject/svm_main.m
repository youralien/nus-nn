load('train_tls.mat') % TODO remove this when submitting
trX = data;
trY = label;

[n_features, n_train] = size(trX);
% normalize to be zero mean, unit variance
mu = mean(trX, 2);
stdev = std(trX, 0, 2);
trX = bsxfun(@rdivide, bsxfun(@minus, trX, mu), stdev);

% SVM fit
margin_type = 'hard';
kernel_type = 'linear';
if margin_type == 'hard'
    C = 10^6; 
else % soft margin
    C = 0.1;
end
if kernel_type == 'linear'
    h1 = trY*trY'; % d_i * d_j
    h2 = trX'*trX; % X_i * X_j
    H = bsxfun(@times, h1, h2); % d_i * d_j * X_i * X_j
else % polynomial kernel
    p = 2;
    h1 = trY*trY'; % d_i * d_j
    h2 = (trX'*trX + 1)^p; % K(X_i, X_j) = Where K is Polynomial
    H = bsxfun(@times, h1, h2); % d_i * d_j * K(X_i, X_j)
end

f = -1 * ones(1, n_train);
Aeq = trY'; % labels
Beq = 0; % alpha_i*d_i = 0
lb = zeros(n_train, 1); % alpha lower bound = 0
ub = C * ones(n_train, 1); % and upper bound = C
alpha0 = [];
options = optimoptions('quadprog','Algorithm','interior-point-convex');
[alpha,fval,exitflag]=quadprog(H,f,[],[],Aeq,Beq,lb,ub,alpha0,options);

sv_idx = find(alpha > graythresh(alpha)); % support vectors are the non-zeroish vectors
w = sum(bsxfun(@times, bsxfun(@times, alpha(sv_idx), trY(sv_idx)), trX(:, sv_idx)'));
random_sv = sv_idx(randperm(length(sv_idx),1));
b = 1 / trY(random_sv) - w*trX(:,random_sv);
% 2D viz
hold on
gscatter(trX(1, :), trX(2, :), trY)
t = -2:0.1:2;
x2 = -1*(w(1)*t+b)/w(2);
plot(t, x2);
hold off

% load test data
load('test_ls.mat');
teX = data;
teY = label;
% normalize test data
teX = bsxfun(@rdivide, bsxfun(@minus, teX, mu), stdev);
