% (bias, input, output)
M = [[1 0.9 8.8],
     [1 1.8 6.2],
     [1 3.0 5.5],
     [1 3.6 2.2],
     [1 4.0 0.7]];
shape = size(M);
X = M(:,1:shape(2)-1);
d = M(:,shape(2));

% standard linear least squares
w = inv(X'*X)*X'*d

% visualize data to be fitted
plot(M(:,2), M(:,3), 'o')
% visualize least squares fit
hold on;
x_range = 0:5;
y_range = w(2)*x_range + w(1);
plot(x_range,y_range, '--');

