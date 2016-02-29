batch_mode = false;

x_train = -2:0.05:2;
x_test = -2:0.01:2;
x_extrap = -3:0.01:3;
y_train = arrayfun(@(x) hw2_q2_func(x), x_train);
y_test = arrayfun(@(x) hw2_q2_func(x), x_test);
y_extrap = arrayfun(@(x) hw2_q2_func(x), x_extrap);

hsize = 10; % hidden layer size
epochs = 50;

for hsize=[7:10 20 50 100]
    train_algo = 'trainbr';
    if batch_mode
        % create and train batch mode net
        net = newff(x_train,y_train,hsize,{'tansig', 'purelin'},train_algo);
        net.trainParam.epochs = epochs;
        net = train(net,x_train,y_train);
    else
        % create and train sequential mode net
        learning_rate = 0.01;
        net = adaptSeq(hsize,x_train,y_train,epochs,learning_rate);
    end

    clf;
    figure(1); % Creates the figure with handle 1
    hFig = figure(1);
    set(gcf,'PaperPositionMode','auto')
    set(hFig, 'Position', [200 200 500 400])

    % show fit on test data
    x_test_pred = sim(net,x_test);
    subplot(2,1,1)
    plot(x_test,y_test,x_test,x_test_pred, 'o')
    title(['MLP 1-', num2str(hsize), '-1 ; Test Data'])
    xlim([-3 3])

    % show extrapolation on values outside of the domain of the input
    x_extrap_pred = sim(net,x_extrap);
    subplot(2,1,2)
    plot(x_extrap,y_extrap,x_extrap,x_extrap_pred, 'o')
    title(['MLP 1-', num2str(hsize), '-1 ; Out-of-training-domain Data'])
    xlim([-3 3])

    if batch_mode
        mode = 'batch';
        train_algo = [train_algo, '_'];
    else
        mode = 'seque';
        train_algo = '';
    end
    saveas(1, ['hw2_q2_', mode, '_', train_algo, num2str(hsize)], 'png')
end