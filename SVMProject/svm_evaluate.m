% load train data
load(strcat(['train.mat'])) % TODO remove this when submitting
trX = data;
trY = label;

% load test data
load(strcat(['test.mat']));
teX = data;
teY = label;

accuracy_table_train = zeros(5,5);
accuracy_table_test = zeros(5,5);

Cs = [0.1 0.6 1.1 2.1 10^6];
ps = [0 2 3 4 5];
kernel_types = {'linear', 'poly', 'poly', 'poly', 'poly'};

for ith=1:length(ps)
    for jth=1:length(Cs)
        svm = SVMEstimator(Cs(jth),kernel_types{ith},ps(ith));
        svm.fit(trX, trY);
        
        subplot(5,5,ith*jth)
        hist(svm.alpha, 100)
        
        pred_train = svm.predict(trX);
        pred_test = svm.predict(teX);
        acc_train = mean(pred_train == trY);
        acc_test = mean(pred_test == teY);
        accuracy_table_train(ith,jth) = acc_train;
        accuracy_table_test(ith,jth) = acc_test;
    end
end

final_accuracy_table = [accuracy_table_train, accuracy_table_test];
