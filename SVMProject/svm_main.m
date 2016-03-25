dataset = '_cls';
load(strcat(['train' dataset '.mat'])) % TODO remove this when submitting
trX = data;
trY = label;

[n_features, n_train] = size(trX);
% normalize to be zero mean, unit variance
mu = mean(trX, 2);
stdev = std(trX, 0, 2);
trX = bsxfun(@rdivide, bsxfun(@minus, trX, mu), stdev);

% -- FIT
margin_type = 'hard';
kernel_type = 'poly';
if strcmp(margin_type,'hard')
    C = 10^6;
else % soft margin
    C = 0.1;
end
if strcmp(kernel_type,'linear')
    h1 = trY*trY'; % d_i * d_j
    K = trX'*trX; % X_i * X_j
    H = bsxfun(@times, h1, K); % d_i * d_j * X_i * X_j
else % polynomial kernel
    p = 2;
    h1 = trY*trY'; % d_i * d_j
    K = (trX'*trX + 1).^p; % K(X_i, X_j) = Where K is Polynomial
    H = bsxfun(@times, h1, K); % d_i * d_j * K(X_i, X_j)
end

f = -1 * ones(1, n_train);
Aeq = trY'; % labels
Beq = 0; % alpha_i*d_i = 0
lb = zeros(n_train, 1); % alpha lower bound = 0
ub = C * ones(n_train, 1); % and upper bound = C
alpha0 = [];
options = optimoptions('quadprog','Algorithm','interior-point-convex');
[alpha,fval,exitflag]=quadprog(H,f,[],[],Aeq,Beq,lb,ub,alpha0,options);

thresh = graythresh(alpha);
sv_idx = find(alpha > thresh); % support vectors are the non-zeroish vectors
w = sum(bsxfun(@times, bsxfun(@times, alpha, trY), trX'));
random_sv = sv_idx(randperm(length(sv_idx),1));
b = 1 / trY(random_sv) - w*trX(:,random_sv); % use support vectors only

% load test data
load(strcat(['test' dataset '.mat']));
teX = data;
teY = label;
% normalize test data
teX = bsxfun(@rdivide, bsxfun(@minus, teX, mu), stdev);

% -- PREDICT
% when calculating g, discriminant function, use support vectors only
Xnew = trX; % what do you want to predict?
if strcmp(kernel_type,'linear')
    Kpred = trX(:,sv_idx)'*Xnew;
else
    Kpred = (trX(:,sv_idx)'*Xnew + 1).^p;
end
g0 = bsxfun(@times, alpha(sv_idx), trY(sv_idx));
g =  g0'*Kpred;
predY = sign(g);
acc = mean(predY == trY')

% 2D viz
if strcmp(dataset,'') == 0 % if its not the full dataset
    clf
    hold on
    gscatter(trX(1, :), trX(2, :), trY)
    if strcmp(kernel_type,'linear')
        t = -2:0.1:2;
        x2 = -1*(w(1)*t+b)/w(2);
        plot(t, x2, 'b--');
    else
        grid_step = 0.03;
        x1 = -2:grid_step:2;
        x2 = -2:grid_step:2;
        [X1,X2] = ndgrid(x1,x2);
        Xbound = [X1(:), X2(:)]';
        if strcmp(kernel_type,'linear')
            Kpred = trX(:,sv_idx)'*Xbound;
        else
            Kpred = (trX(:,sv_idx)'*Xbound + 1).^p;
        end
        g0 = bsxfun(@times, alpha(sv_idx), trY(sv_idx));
        g =  g0'*Kpred;
        bound_idx = find(abs(g) < 0.05);
        plot(Xbound(1, bound_idx), Xbound(2,bound_idx), 'b.')
    end
    hold off
end

