clf

batch_mode = false;

x_train = -2:0.05:2;
x_test = -2:0.01:2;
y_train = arrayfun(@(x) hw2_q2_func(x), x_train);
y_test = arrayfun(@(x) hw2_q2_func(x), x_test);

hsize = 10; % hidden layer size
epochs = 50;

if batch_mode
    % create and train batch mode net
    net = newff(x_train,y_train,hsize,{'tansig', 'purelin'},'trainlm');
    net.trainParam.epochs = epochs;
    net = train(net,x_train,y_train);
else
    % create and train sequential mode net
    learning_rate = 0.001;
    net = adaptSeq(hsize,x_train,y_train,epochs,learning_rate);
end

% show fit on training data
x_train_pred = sim(net,x_train);
plot(x_train,y_train,x_train,x_train_pred, 'o')

x_test_pred = sim(net,x_test);
plot(x_test,y_test,x_test,x_test_pred, 'o')