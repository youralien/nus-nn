# NUS EE5904 NEURAL NETWORKS HW2 Report
by Ryan Louie, A0149643X

## Q1: a)
Using gradient descent, with a `learning rate = 0.001`, we defined convergence to the minimum where `f(x,y) = 0` when the function value was within `epsilon = 0.01` of 0.  As depicted below, the learning converges after 3099 iterations.

![Gradient Descent, lr=0.001: Weight Trajectory and Function Value over learning](hw2_q1_gradientdescent.png){height=300px}

Using a learning rate that is too large (`learning rate = 0.1`) results in the learning diverging, where the function value is no longer decreasing over iterations.

![Gradient Descent, lr=0.1: Weight Trajectory and Function Value over learning](hw2_q1_gradientdescent_learning_rate_too_high.png){height=300px}

## Q1: b)
Newton's method is a huge improvement over gradient descent.  The learning converges in as little as 7 iterations. Looking at the top axes describing the change in weights in the 2-D space, we can see that large jumps in `(X,Y)` are achieved that progress closer to where the function minimum is located at `(1,1)`,

![Newton's Method: Weight Trajectory and Function Value over learning](hw2_q1_newton.png){height=300px}

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
Since the last number of my ID equals 3, I will be working with the Gender Classification task.  For the binary labels, `{Male: 1, Female: 0}`.

Train Set: 1000 Examples
Negative Examples: 0.261000
Postive  Examples: 0.739000

Test Set: 250 Examples
Negative Examples: 0.264000
Postive  Examples: 0.736000

## Q3: training notes
Before discussing the performance metrics for gender classification, using the various network architectures and learning procedures, I want to talk about some important preprocessing used on the data which was given to all the models.

Most of my early experiments involved taking the recommendations from the notes:

1. normalize each of the features over the dataset by computing the zscore per-feature.

2. adjusted the targets to be `{0.2, 0.8}`, values inside the range of the sigmoid output activation function

However I was still unable to achieve satisfactory test accuracies with this scheme.  When I visualized these zscore features, I thought I might have discovered my problem.

![Face using zscore normalization on each feature, where `mu`
 and `sigma` are computed across the dataset](nus-nnet-hw2-faces/example_face_zscore.png){height=250px}

While the current strategy was making my features in the viable range for the tanh activation units to operate properly, the image itself lost lots of its visual meaning!  As an alternative, I read about an idea to zscore over a single example image, an operation that corresponds to contrast normalization.

![Face using zscore normalization on the entire image, where mu and sigma are computed over the pixel values of a single example image](nus-nnet-hw2-faces/example_face_contrast.png){height=250px}

The pixel intensities of this visualization seem much more representative of a human face, while still benefiting from having the values in the range for the activation units to operate properly.

## Q3: b)
The perceptron used a learning rate of 0.001.

Train Error: 0.00; Test Error: 0.16

![Perceptron learning on per-example contrast normalized data](nus-nnet-hw2-faces/contrast_perf_perceptron.png){height=300px}


## Q3: c)
For SGD sequential learning, the following hyperparameters were used:

- learning_rate=0.0005
- epochs=35

Even with the small learning rate, the cost over epochs fluctated, as expected.  In addition, we typically saw training error convergence within 30 epochs; any longer, and the test error would continually get worse.  Choosing the maximum epochs to be relatively low served as an early stopping strategy.

Train Error: 0.00; Test Error: 0.15

![MLP with sequential mode learning on per-example contrast normalized data](nus-nnet-hw2-faces/contrast_perf_mlp_seque.png){height=300px}

## Q3: d)
For SGD batch learning, the following hyperparameters were used:
- learning_rate=0.05
- epochs=150

Train Error: 0.00; Test Error: 0.09

![MLP with batch mode learning on per-example contrast normalized data](nus-nnet-hw2-faces/contrast_perf_mlp_batch.png){height=300px}

## Q3: e)
The importance of having the eyes of each face centered at the same place cannot be overstated. Having the same pixels in the image across the dataset correspond to the eyes means that the model can reliably learn patterns from these eye features.  In addition, other parts of the face like glasses, eye brows, and cheek bones are spatially close to the eyes, and likely will benefit similarly in the fixing of the pixel positions for these important facial characteristics.

## Q3: f)
![Examples which the MLP has misclassified](nus-nnet-hw2-faces/failures/allfailures.png)

From the examples of faces that were misclassified on gender, I observed the following things:

- Faces were not centered on the image (even though the eyes were, we had a mix of front and side profile)
- Shadows
- Faces zoomed in that do not have hair
- Backgrounds walls behind the face that can be confused for hair

These observations suggest that perhaps, the model is very sensitive to shadows in the image, and that perhaps the model is learning features about hair, which is not always present in the test set.

Frankly, it is difficult to compare, using simple visual inspection, the distribution of correct and misclassified examples. Nonetheless, my suggestions for improvement are given:

- Perform transformations on the image training data, so there is less potential for the model to overfit on nuances in the data.  For examples, if we flip the image across its vertical axis before flattening the pixel features for the MLP, we can essentially double our training data while remaining more invariant to faces that feature side profiles. Not that this may affect the alignment of the eyes across images, which could have an opposite detrimental effect.
- Use the labels for the other classification tasks, even though we are trying to learn gender.  To set up a network like this, we would have 3 output nodes (gender, glasses, smiles), and we would train the additional labels.  What we can hope from this is that the same network weights will be forced to learn more general features about faces, that will eventually help to increase the performance on the test set for gender, as well as the other tasks.
- Use resizing of the image (using interpolation) to create a coarser resolution image, which with the interpolations, may be less sensitive to noise in the many pixel features.  We can then use these coarser model features, and feed it another network of the same structure. Then we can take an ensemble of the probabilistic outputs.

## Q3: code
The code for question 3 was implemented in Python with the Theano library. In the interest of printing only the relevant portions, I've included the scripts, without the source code for their imported modules.  All source code for the faces assignment can be found at [](https://github.com/youralien/nus-nn/tree/master/nus-nnet-hw2-faces)

###Code for the single layer perceptron:

    import numpy as np
    import theano
    from theano import tensor as T
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    from sklearn.metrics import accuracy_score
    from faces import faces, permute
    from performanceplot import performanceplot

    srng = RandomStreams()

    trX, trY, teX, teY = faces(contrast=True, perceptron=True)
    print "target values (min, max): ", (teY.min(), teY.max())
    print trX.shape
    print trY.shape
    input_dim = trX.shape[1]


    def floatX(X):
        return np.asarray(X, dtype=theano.config.floatX)

    def init_weights(shape):
        return theano.shared(floatX(np.random.randn(*shape) * 0.02))

    def model(X, w_o):
        return T.sgn(T.dot(X, w_o)) 

    X = T.fmatrix()
    Y = T.fmatrix()

    # theano.config.compute_test_value = 'warn' # Use 'warn' to activate this featureg
    # X.tag.test_value = np.zeros((1, input_dim), dtype='float32')
    # Y.tag.test_value = np.zeros((1, 1), dtype='float32')

    w_o = init_weights((input_dim, 1))

    y_pred = model(X, w_o)

    batch_size=1; learning_rate=0.001; # sequential mode: single example

    cost = T.mean(Y - y_pred)
    update = [[w_o, w_o + learning_rate*cost*T.transpose(X)]]

    train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

    print "batch_size: ", batch_size
    print "learning_rate: ", learning_rate

    cost_record = []
    train_error_record = []
    test_error_record = []
    for epoch in range(100):
        if isinstance(batch_size, int):
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                cost = train(trX[start:end], trY[start:end])
        else:
            cost = train(trX, trY)
        cost_record.append(cost)
        if epoch % 1 == 0:
            train_error = 1-np.mean(np.sign(trY)== predict(trX))
            test_error = 1-np.mean(np.sign(teY)== predict(teX))
            train_error_record.append(train_error)
            test_error_record.append(test_error)
            print "%d,%0.4f,%0.4f" % (epoch, train_error, test_error)
            trX, trY = permute(trX, trY)

    performanceplot(cost_record, train_error_record, test_error_record, 'contrast_perf_perceptron.png')

###Code for the MLP

    import numpy as np
    import theano
    from theano import tensor as T
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    from sklearn.metrics import accuracy_score
    from faces import faces, permute
    from performanceplot import performanceplot
    import failure_analysis

    srng = RandomStreams()

    trX, trY, teX, teY = faces(zscore=False, onehot=False, adjust_targets=True, contrast=True)
    print trX.shape
    print trY.shape
    print teX.shape
    print teY.shape
    input_dim = trX.shape[1]


    def floatX(X):
        return np.asarray(X, dtype=theano.config.floatX)

    def init_weights(shape):
        return theano.shared(floatX(np.random.randn(*shape) * 0.02))

    def sgd(cost, params, lr=0.05):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            updates.append([p, p - g * lr])
        return updates

    def dropout(X, p=0.):
        if p > 0:
            retain_prob = 1 - p
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X

    def model(X, w_h, w_o, p_drop_input, p_drop_hidden):
        X = dropout(X, p_drop_input)
        h = T.tanh(T.dot(X, w_h))
        
        h = dropout(h, p_drop_hidden)
        pyx = T.nnet.sigmoid(T.dot(h, w_o))
        return pyx, h

    X = T.fmatrix()
    Y = T.fmatrix()

    h1_size = 75 
    w_h1 = init_weights((input_dim, h1_size))
    w_o = init_weights((h1_size, 1))

    # p_dropout = 0.5 # healthy amounts of dropout
    p_dropout = 0.  # no drop out
    py_x, h1 = model(X, w_h1, w_o, p_dropout, p_dropout)
    y_proba, h1 = model(X, w_h1, w_o, 0., 0.)
    y_pred = y_proba > 0.5

    # -- learning rate is coupled with batch size!
    batch_size=''; learning_rate=0.05; epochs=150; # batch mode: entire batch
    # batch_size=1; learning_rate=0.0005; epochs=35; # sequential mode: single example
    # batch_size=20; learning_rate=0.05; epochs=35;# minibatches good for SGD, like sequential

    cost = T.mean(T.nnet.binary_crossentropy(py_x, Y))
    params = [w_h1, w_o]
    update = sgd(cost, params, lr=learning_rate) 

    train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)
    compute_H = theano.function(inputs=[X], outputs=h1, allow_input_downcast=True)

    print "batch_size: ", batch_size
    print "learning_rate: ", learning_rate
    print "p_dropout", p_dropout

    cost_record = []
    tr_err_record = []
    te_err_record = []
    for i in range(epochs):
        if isinstance(batch_size, int):
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                cost = train(trX[start:end], trY[start:end])
        else:
            cost = train(trX, trY)
        cost_record.append(cost)
        tr_err = 1-np.mean((trY > 0.5) == predict(trX))
        te_err = 1-np.mean((teY > 0.5) == predict(teX))
        tr_err_record.append(tr_err)
        te_err_record.append(te_err)
        print "%d,%0.4f,%0.4f" % (i, tr_err, te_err)
        trX, trY = permute(trX, trY)

    if isinstance(batch_size, int):
        if batch_size == 1:
            fig_outfile = 'perf_mlp_seque.png'
        else:
            fig_outfile = 'perf_mlp_minibatchsize_%d.png' % batch_size
    else:
        fig_outfile = 'perf_mlp_batch.png'
    performanceplot(cost_record, tr_err_record, te_err_record, "contrast_" + fig_outfile)
    failure_analysis.investigate_mlp(teX, teY, predict(teX) > 0.5)

    H = compute_H(trX)
    _0 , svals, _1 = np.linalg.svd(H)

    def compute_effective_rank(svals, gamma=0.99):
        effective_rank = 1
        for k in range(1, svals.shape[0]):
            if np.sum(svals[:k]) / np.sum(svals) >= gamma:
                break
            else:
                effective_rank += 1
        return effective_rank

    print "Rank: {}, Hidden Layer Size: {}".format(compute_effective_rank(svals), svals.shape[0])