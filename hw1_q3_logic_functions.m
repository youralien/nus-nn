% hyperparameters
learning_rate = 1;
num_steps = 50;

% make default plot style favor symbols rather than Color
m = {'.','o','*','x','s','d','^','v','>','<','p','h'};
set_marker_order = @() set(gca(), ...
    'LineStyleOrder',m, 'ColorOrder',[0 0 0], ...
    'NextPlot','replacechildren');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AND
% columns = (bias, x1, x2, y)
M = [[1 0 0 0],
     [1 0 1 0],
     [1 1 0 0],
     [1 1 1 1]];

W = perceptron_fit(M, learning_rate, num_steps, true);

% plot the weight trajectories
subplot(2,2,1);
set_marker_order();
plot(W);
title('Learned weights of the AND problem');
legend('b','w1','w2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OR
% columns = (bias, x1, x2, y)
M = [[1 0 0 0],
     [1 0 1 1],
     [1 1 0 1],
     [1 1 1 1]];

W = perceptron_fit(M, learning_rate, num_steps, true);

% plot the weight trajectories
subplot(2,2,2);
set_marker_order();
plot(W);
title('Learned weights of the OR problem');
legend('b','w1','w2');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPLEMENT
% columns = (bias, x1, y)
M = [[1 0 1],
     [1 1 0]];

W = perceptron_fit(M, learning_rate, num_steps, true);

% plot the weight trajectories
subplot(2,2,3);
set_marker_order();
plot(W);
title('Learned weights of the COMPLEMENT problem');
legend('b','w1');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% XOR
% columns = (bias, x1, x2, y)
M = [[1 0 0 0],
     [1 0 1 1],
     [1 1 0 1],
     [1 1 1 0]];

W = perceptron_fit(M, learning_rate, num_steps, true);

% plot the weight trajectories
subplot(2,2,4);
set_marker_order();
plot(W);
title('Learned weights of the XOR problem');
legend('b','w1','w2');
