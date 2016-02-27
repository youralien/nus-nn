function g = rosenbrock_valley_gradient(x,y)
    % gradient equations found at
    % http://www.mme.wsu.edu/~grantham/papers/TF/cdc03_frm111.pdf
    a = 100;
    g = zeros(1,2);
    g(1,1) = 4*a*x*(x^2-y) + 2*(x-1);
    g(1,2) = -2*a*(x^2 - y);
end
