% make default plot style favor symbols rather than Color
m = {'.','o','*','x','s','d','^','v','>','<','p','h'};
set_marker_order = @() set(gca(), ...
    'LineStyleOrder',m, 'ColorOrder',[0 0 0], ...
    'NextPlot','replacechildren');

% (bias, input, output)
M = [[1 0.9 8.8],
     [1 1.8 6.2],
     [1 3.0 5.5],
     [1 3.6 2.2],
     [1 4.0 0.7]];
shape = size(M);

learning_rate = 0.02;
epochs = 100;
n_steps = epochs * shape(1);
[W, error] = LMS_fit(M, learning_rate, n_steps, true);

% plot the weight trajectories
subplot(1,2,1);
% set_marker_order();
plot(W);
title('Learned weights for input-output problem');
legend('b','w1');

% visualize data to be fitted
subplot(1,2,2);
plot(M(:,2), M(:,3), 'o')
% visualize least squares fit
hold on;

x_range = 0:5;
slope = W(n_steps + 1, 2)
intercept = W(n_steps + 1, 1)
y_range = slope*x_range + intercept;
plot(x_range,y_range, '--');
title('LMS fitting result')
