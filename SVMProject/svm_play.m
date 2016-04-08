%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question 1 SVM on Toy Data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datasets = {'_tls', '_cls'};
Cs = [10^6, 0.6];
kernel_types = {'linear', 'poly'};
for idx=1:length(datasets)
    dataset = datasets{idx};
    % load train data
    load(strcat(['train' dataset '.mat'])) % TODO remove this when submitting
    trX = data;
    trY = label;

    % load test data
    load(strcat(['test' dataset '.mat']));
    teX = data;
    teY = label;

    svm = SVMEstimator(Cs(idx),kernel_types{idx},3, 1e-84);
    svm.fit(trX, trY);
    pred_train = svm.predict(trX);
    pred_test = svm.predict(teX);
    acc_train = mean(pred_train == trY)
    acc_test = mean(pred_test == teY)

    subplot(1,length(datasets),idx)
    visualize_toydata(trX,trY,svm)
end