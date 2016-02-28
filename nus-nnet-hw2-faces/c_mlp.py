import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.metrics import accuracy_score
from faces import faces, permute

srng = RandomStreams()

trX, trY, teX, teY = faces(zscore=True, onehot=False, adjust_targets=False)
print trX.shape
print trY.shape
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

h_size = 128
w_h = init_weights((input_dim, h_size))
w_o = init_weights((h_size, 1))

py_x, h = model(X, w_h, w_o, 0., 0.)
y_proba, h = model(X, w_h, w_o, 0., 0.)
y_pred = y_proba > 0.5

# -- learning rate is coupled with batch size!
# batch_size=''; learning_rate=0.05; # batch mode: entire batch
batch_size=1; learning_rate=0.0005; # sequential mode: single example
# batch_size=80; learning_rate=0.01; # minibatches

cost = T.mean(T.nnet.binary_crossentropy(py_x, Y))
params = [w_h, w_o]
update = sgd(cost, params, lr=learning_rate) 

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)
compute_H = theano.function(inputs=[X], outputs=h, allow_input_downcast=True)

for i in range(100):
    if isinstance(batch_size, int):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            cost = train(trX[start:end], trY[start:end])
    else:
        cost = train(trX, trY)
    if i % 1 == 0:
        print i, 1-np.mean(trY == predict(trX)), 1-np.mean(teY == predict(teX))
        trX, trY = permute(trX, trY)
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

