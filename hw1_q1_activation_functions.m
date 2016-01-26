syms x1 x2;
w1 = 1;
w2 = 1;
b = 0;
v = w1*x1 + w2*x2 + b;
thresh = 0.5;
range = [-pi,pi];

% activation: pure linear function
subplot(3,1,1);
ezplot(v == thresh, range);

% activation: hyperbolic tangent function
subplot(3,1,2);
ezplot((1 - exp(-v)) / (1 + exp(-v)) == thresh, range);

% activation: gaussian function
subplot(3,1,3);
ezplot(exp(-(v^2)/2) == thresh, range);
