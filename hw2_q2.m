x_train = -2:0.05:2;
x_test = -2:0.01:2;
y_train = arrayfun(@(x) hw2_q2_func(x), x_train);
y_test = arrayfun(@(x) hw2_q2_func(x), x_test);

hsize = 10; % hidden layer size
net = newff(x_train,y_train,hsize,{'tansig', 'purelin'},'trainlm');

clf
plot(x_train,y_train, '.')