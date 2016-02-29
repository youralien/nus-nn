# NUS EE5904 NEURAL NETWORKS HW2 Report
by Ryan Louie, A0149643X

## Q1: a)
Using gradient descent, with a `learning rate = 0.001`, we defined convergence to the minimum where `f(x,y) = 0` when the function value was within `epsilon = 0.01` of 0.  As depicted below, the learning converges after 3099 iterations.
![Gradient Descent, lr=0.001: Weight Trajectory and Function Value over learning](hw2_q1_gradientdescent.png)

Using a learning rate that is too large (`learning rate = 0.1`) results in the learning diverging, where the function value is no longer decreasing over iterations.
![Gradient Descent, lr=0.1: Weight Trajectory and Function Value over learning](hw2_q1_gradientdescent_learning_rate_too_high.png)

## Q1: b)
Newton's method is a huge improvement over gradient descent.  The learning converges in as little as 7 iterations. Looking at the top axes describing the change in weights in the 2-D space, we can see that large jumps in `(X,Y)` are achieved that progress closer to where the function minimum is located at `(1,1)`,
![Newton's Method: Weight Trajectory and Function Value over learning](hw2_q1_newton.png)

## Q1: code
    % part a: steepest gradient descent
    lr = 0.001;
    runway = 10^6;
    epsilon = 0.01;
    w = rand(runway, 2); % weights are also input x,y
    out = zeros(runway,1);
    iter = 1;
    newton = true; # newtons method or not; toggle for part a) or part b)

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

The functions to compute the function value, gradient, and hessian of Rosenbrock's Valley are defined as follows

    function out = rosenbrock_valley(x,y)
        out = (1 - x)^2 + 100*(y-x^2)^2;
    end

    function g = rosenbrock_valley_gradient(x,y)
        % gradient equations found at
        % http://www.mme.wsu.edu/~grantham/papers/TF/cdc03_frm111.pdf
        a = 100;
        g = zeros(1,2);
        g(1,1) = 4*a*x*(x^2-y) + 2*(x-1);
        g(1,2) = -2*a*(x^2 - y);
    end

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


## Q2: a)
The following figures show different MLP architectures of the form `1-n-1`, trained using sequential mode of gradient-descent learning, to learn the particular sinusoid function.  The following can be observed:

* As `n = 10`, the test data shows near perfect fitting.
* When `n < 10`, the test data is underfit. The predicted values do not completely match the higher frequency sinusoids of the function
* When `n > 10`, overfitting occurs.  It is especially severe for `n = 50` or `n = 100`, where the predicted values are finding features in the function that are not there.
* For all cases, extrapolation in the range `[-3, -2) U (2, 3)` is poor, where there were no training examples for the network to learn from.

![](hw2_q2_seque_1.png){height=200px}
![](hw2_q2_seque_2.png){height=200px}
![](hw2_q2_seque_3.png){height=200px}
![](hw2_q2_seque_4.png){height=200px}
![](hw2_q2_seque_5.png){height=200px}
![](hw2_q2_seque_6.png){height=200px}
![](hw2_q2_seque_7.png){height=200px}
![](hw2_q2_seque_8.png){height=200px}
![](hw2_q2_seque_9.png){height=200px}
![](hw2_q2_seque_10.png){height=200px}
![](hw2_q2_seque_20.png){height=200px}
![](hw2_q2_seque_50.png){height=200px}
![](hw2_q2_seque_100.png){height=200px}

## Q2: b)
The same training and visualization procedure is performed, with a change using the method of batch learning with the `trainlm` rule.  The following was observed:

* `n = 7` has a suprisingly good fit, despite not have the minimum number of hidden units for full representation
* `n = 8` underfits more than `n = 7`, which might be suprising since more nodes meant more fine grained representation using sequential learning
* `n = 10` fits well, but it isn't as good of a fit as `n = 7` or `n = 9`!  This architecture is supposed to be the perfect size for the dataset, according to our analysis, and yet it does not perform to expectations
* The architectures where `n > 10` have spurious parts in the function which does not follow the actual shape of the function.  Visually, it overfits as poorly as sequential learning.
* Again, all extrapolation in regions of the domain the network has not seen examples show poor fitting.

![](hw2_q2_batch_trainlm_1.png){height=200px}
![](hw2_q2_batch_trainlm_2.png){height=200px}
![](hw2_q2_batch_trainlm_3.png){height=200px}
![](hw2_q2_batch_trainlm_4.png){height=200px}
![](hw2_q2_batch_trainlm_5.png){height=200px}
![](hw2_q2_batch_trainlm_6.png){height=200px}
![](hw2_q2_batch_trainlm_7.png){height=200px}
![](hw2_q2_batch_trainlm_8.png){height=200px}
![](hw2_q2_batch_trainlm_9.png){height=200px}
![](hw2_q2_batch_trainlm_10.png){height=200px}
![](hw2_q2_batch_trainlm_20.png){height=200px}
![](hw2_q2_batch_trainlm_50.png){height=200px}
![](hw2_q2_batch_trainlm_100.png){height=200px}

## Q2: c)
The same training and visualization procedure is performed, with a change using the method of batch learning with the `trainbr` rule.  The following was observed:

* The `1-6-1` MLP underfits by a lot.  Compared to the `sequential` and `trainlm` learning procedures for the same network topology, the fit due to regularized back prop is a straight line and does not capture any of the sinusoidal characteristics of the function
* Network architectures for `n = 7, 8, 9, 10` all have near perfect fits, which is not unexpected given similar results for part a) and b)
* The most suprising is for the topologies with `n = 50, 100`. Under the unregularized modes of training, these networks had overfit badly, producing functions with spurious characteristics.  However, baysian regularized back prop allows these network architectures to maintain their perfect fit.

![](hw2_q2_batch_trainbr_1.png){height=200px}
![](hw2_q2_batch_trainbr_2.png){height=200px}
![](hw2_q2_batch_trainbr_3.png){height=200px}
![](hw2_q2_batch_trainbr_4.png){height=200px}
![](hw2_q2_batch_trainbr_5.png){height=200px}
![](hw2_q2_batch_trainbr_6.png){height=200px}
![](hw2_q2_batch_trainbr_7.png){height=200px}
![](hw2_q2_batch_trainbr_8.png){height=200px}
![](hw2_q2_batch_trainbr_9.png){height=200px}
![](hw2_q2_batch_trainbr_10.png){height=200px}
![](hw2_q2_batch_trainbr_20.png){height=200px}
![](hw2_q2_batch_trainbr_50.png){height=200px}
![](hw2_q2_batch_trainbr_100.png){height=200px}

## Q2: code
To perform sequential learning, the given `adaptSeq` function was used.

The sinusoidal function we were drawing samples from for data is given below:

    function y = hw2_q2_func(x)
        y = 1.2*sin(pi*x) - cos(2.4*pi*x);
    end

And the rest of the code which trained the networks and plotted their fitting results is given below:

    batch_mode = false;

    x_train = -2:0.05:2;
    x_test = -2:0.01:2;
    x_extrap = -3:0.01:3;
    y_train = arrayfun(@(x) hw2_q2_func(x), x_train);
    y_test = arrayfun(@(x) hw2_q2_func(x), x_test);
    y_extrap = arrayfun(@(x) hw2_q2_func(x), x_extrap);

    hsize = 10; % hidden layer size
    epochs = 50;

    for hsize=[7:10 20 50 100]
        train_algo = 'trainbr';
        if batch_mode
            % create and train batch mode net
            net = newff(x_train,y_train,hsize,{'tansig', 'purelin'},train_algo);
            net.trainParam.epochs = epochs;
            net = train(net,x_train,y_train);
        else
            % create and train sequential mode net
            learning_rate = 0.01;
            net = adaptSeq(hsize,x_train,y_train,epochs,learning_rate);
        end

        clf;
        figure(1); % Creates the figure with handle 1
        hFig = figure(1);
        set(gcf,'PaperPositionMode','auto')
        set(hFig, 'Position', [200 200 500 400])

        % show fit on test data
        x_test_pred = sim(net,x_test);
        subplot(2,1,1)
        plot(x_test,y_test,x_test,x_test_pred, 'o')
        title(['MLP 1-', num2str(hsize), '-1 ; Test Data'])
        xlim([-3 3])

        % show extrapolation on values outside of the domain of the input
        x_extrap_pred = sim(net,x_extrap);
        subplot(2,1,2)
        plot(x_extrap,y_extrap,x_extrap,x_extrap_pred, 'o')
        title(['MLP 1-', num2str(hsize), '-1 ; Out-of-training-domain Data'])
        xlim([-3 3])

        if batch_mode
            mode = 'batch';
            train_algo = [train_algo, '_'];
        else
            mode = 'seque';
            train_algo = '';
        end
        saveas(1, ['hw2_q2_', mode, '_', train_algo, num2str(hsize)], 'png')
    end

## Q3: a)
## Q3: b)
## Q3: c)
## Q3: d)
## Q3: e)
## Q3: f)


