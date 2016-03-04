import numpy as np
import theano
from six.moves import cPickle
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

mode = raw_input("What should this script do? (train, something else):")

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

def model2layer(X, w_h1, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h1 = T.tanh(T.dot(X, w_h1))
    h1 = dropout(h1, p_drop_hidden)

    h2 = T.tanh(T.dot(h1, w_h2))
    h2 = dropout(h2, p_drop_hidden)
    pyx = T.nnet.sigmoid(T.dot(h2, w_o))
    return pyx, h1, h2

def model1layer(X, w_h1, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h1 = T.tanh(T.dot(X, w_h1))
    h1 = dropout(h1, p_drop_hidden)
    pyx = T.nnet.sigmoid(T.dot(h1, w_o))
    return pyx, h1

n_layers = int(raw_input("How many layers? 1 or 2:"))
if mode == 'train':
    X = T.fmatrix()
    Y = T.fmatrix()

    h1_size = 75 
    w_h1 = init_weights((input_dim, h1_size))
    if n_layers == 2:
        w_h2 = init_weights((h1_size, h1_size))
    w_o = init_weights((h1_size, 1))

    # p_dropout = 0.5 # healthy amounts of dropout
    p_dropout = 0.5  # no drop out
    if n_layers == 2:
        py_x, h1, h2 = model2layer(X, w_h1, w_h2, w_o, p_dropout, p_dropout)
        y_proba, h1, h2 = model2layer(X, w_h1, w_h2, w_o, 0., 0.)
    elif n_layers == 1:
        py_x, h1 = model1layer(X, w_h1, w_o, p_dropout, p_dropout)
        y_proba, h1 = model1layer(X, w_h1, w_o, 0., 0.)

    y_pred = y_proba > 0.5

    # -- learning rate is coupled with batch size!
    # batch_size=''; learning_rate=0.05; epochs=150; # batch mode: entire batch
    # batch_size=1; learning_rate=0.0005; epochs=35; # sequential mode: single example
    batch_size=50; learning_rate=0.01; epochs=100;# minibatches good for SGD, like sequential

    cost = T.mean(T.nnet.binary_crossentropy(py_x, Y))
    if n_layers == 1:
        params = [w_h1, w_o]
    elif n_layers == 2:
        params = [w_h1, w_h2, w_o]
    update = sgd(cost, params, lr=learning_rate) 

    train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)
    compute_H1 = theano.function(inputs=[X], outputs=h1, allow_input_downcast=True)
    if n_layers == 2:
        compute_H2 = theano.function(inputs=[X], outputs=h2, allow_input_downcast=True)

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

    # Computing Effective Rank for Model Architecture
    H1 = compute_H1(trX)
    H2 = compute_H2(trX)

    for H in [H1, H2]:
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

    if isinstance(batch_size, int):
        if batch_size == 1:
            fig_outfile = 'perf_mlp_seque.png'
        else:
            fig_outfile = 'perf_mlp_minibatchsize_%d.png' % batch_size
    else:
        fig_outfile = 'perf_mlp_batch.png'

    if raw_input("Shall we save this model? (y/n)\n") == 'y':
        model_outfile = fig_outfile.split('.')[0] + ".pkl"
        fobj = open(model_outfile, 'wb')
        cPickle.dump(predict, fobj, protocol=cPickle.HIGHEST_PROTOCOL)
        fobj.close()

    if raw_input('Save training figure? (y/n): \n') == 'y':
        performanceplot(cost_record, tr_err_record, te_err_record, "contrast_" + fig_outfile)

else:
    model_outfile = raw_input("Provide path to model_outfile: \n")
    fobj = open(model_outfile, 'rb')
    cPickle.load(fobj)
    fobj.close()

if raw_input("Perform failure analysis? (y/n):\n") == 'y':
    failure_analysis.investigate_mlp(teX, teY, predict(teX) > 0.5)

