function H = rosenbrock_valley_hessian(x,y)
    % hessian equations found at
    % http://www.mme.wsu.edu/~grantham/papers/TF/cdc03_frm111.pdf
    a = 100;
    H = zeros(2,2);
    H(1,1) = 12*a*x^2 - 4*a*y + 2;
    H(1,2) = -4*a*x;
    H(2,1) = -4*a*x;
    H(2,2) = 2*a;
end
