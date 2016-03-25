function [train, test] = kfoldcrossval(N, K)
    idxs = randperm(N);
    division = 1 / K * N;
    test = idxs(1:division-1);
    train = idxs(division:length(idxs));
end