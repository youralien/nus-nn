% (bias, input, output)
M = [[1 0.9 8.8],
     [1 1.8 6.2],
     [1 3.0 5.5],
     [1 3.6 2.2],
     [1 4.0 0.7]];
shape = size(M);

%% LMS %%
learning_rate = 0.02;
epochs = 100;
n_steps = epochs * shape(1);
[W, error] = LMS_fit(M, learning_rate, n_steps, true);

% plot the weight trajectories
subplot(1,2,1);
plot(W);
title('Learned weights for input-output problem');
xlabel('Number of examples seen')
ylabel('Weight/Bias Value')
legend('b','w1', 'Location', 'northwest');


% visualize data to be fitted
subplot(1,2,2);
plot(M(:,2), M(:,3), 'o')
% visualize least mean squares fit
hold on;
x_range = 0:5;
slope = W(n_steps + 1, 2)
intercept = W(n_steps + 1, 1)
y_range = slope*x_range + intercept;
plot(x_range,y_range, '--');
title('LMS fitting result')
xlabel('Input')
ylabel('Output')

%% Linear Least Squares %%
X = M(:,1:shape(2)-1);
d = M(:,shape(2));

w = inv(X'*X)*X'*d
subplot(1,2,2);
hold on;
y_range1 = w(2)*x_range + w(1);
plot(x_range,y_range1, '-.');
legend('data', 'LMS', 'LLS');