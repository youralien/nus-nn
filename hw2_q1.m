% part a: steepest gradient descent
lr = 0.001;
runway = 10^6;
epsilon = 0.01;
w = rand(runway, 2); % weights are also input x,y
out = zeros(runway,1);
iter = 1;
newton = true;

% first out
out(iter,1) = rosenbrock_valley(w(iter,1), w(iter,2));
while out(iter,1) > epsilon
    if newton
        H = rosenbrock_valley_hessian(w(iter,1), w(iter,2));
        g = rosenbrock_valley_gradient(w(iter,1), w(iter,2));
        delta_w = (-inv(H)*g')';
    else
        % gradient descent
        g = rosenbrock_valley_gradient(w(iter,1), w(iter,2));
        delta_w = -lr*g;
    end
    
    w(iter+1,:) = w(iter,:) + delta_w;
    iter = iter + 1;
    out(iter,1) = rosenbrock_valley(w(iter,1), w(iter,2));
end

% how many iterations
iter
% how close of a solution
solution = w(iter,:)

clf;
subplot(2,1,1)
plot(w(1:iter,1), w(1:iter,2))
xlabel('Input X')
ylabel('Input Y')

subplot(2,1,2)
semilogy(out(1:iter))
xlabel('Iterations')
ylabel('Function Value (log)')
ylim([0 100])