% load train data
load(strcat(['train.mat'])) % data and label exist
% shuffle data
shuffle_idx = randperm(length(label));
data = data(:,shuffle_idx);
label = label(shuffle_idx);
n_examples = length(label);

accuracy_table_train = zeros(5,5);
accuracy_table_valid = zeros(5,5);

Cs = [0.1 0.6 1.1 2.1 10^6];
ps = [0 2 3 4 5];
kernel_types = {'linear', 'poly', 'poly', 'poly', 'poly'};

% results in 67% train, 33% test
% similar to the 65% train, 35% test introduced by 285 : 100 ratio
k_folds = 3;

for ith=1:length(ps)
    for jth=1:length(Cs)
        acc_train = zeros(k_folds,1);
        acc_valid = zeros(k_folds,1);
        for fold=1:k_folds % 3-fold validation
            [train_idx, valid_idx] = kfoldcrossval(n_examples,k_folds);
            trX = data(:, train_idx);
            trY = label(train_idx);
            vaX = data(:, valid_idx);
            vaY = label(valid_idx);
            
            svm = SVMEstimator(Cs(jth),kernel_types{ith},ps(ith),1e-10);
            svm.fit(trX, trY);
            
            subplot(5,5,ith*jth)
            hist(svm.alpha, 100)
            
            pred_train = svm.predict(trX);
            pred_valid = svm.predict(vaX);
            acc_train(fold) = mean(pred_train == trY);
            acc_valid(fold) = mean(pred_valid == vaY);
            
        end
        accuracy_table_train(ith,jth) = mean(acc_train);
        accuracy_table_valid(ith,jth) = mean(acc_valid);
    end
end

final_accuracy_table = [accuracy_table_train, accuracy_table_valid];
