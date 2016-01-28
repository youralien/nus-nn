function [W, error] = LMS_fit(M, lr, n_steps, zero_bias)

    shape = size(M);
    
    % initialize weights, bias included
    W = rand(n_steps + 1,shape(2)-1);
    if zero_bias
        % initialize bias to be zero
        W(:, 1) = zeros(n_steps + 1, 1);
    end

    
    X = M(:,1:shape(2)-1);
    ytrue = M(:,shape(2));
    ypred = zeros(shape(1),1);
    error = zeros(n_steps,1);

    % learning procedure
    for step = 1:n_steps
        row = mod(step, shape(1)) + 1;

        % perceptron prediction
        ypred(row) = X(row, :)*W(step,:)'; 
        error(step) = ytrue(row) - ypred(row);

        % weight update
        W(step+1, :) = W(step,:) + lr*error(step)*X(row,:);
    end

end
