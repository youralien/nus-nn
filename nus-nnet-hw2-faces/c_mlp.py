import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.metrics import accuracy_score
from faces import faces, permute

srng = RandomStreams()

trX, trY, teX, teY = faces(zscore=True, onehot=False, adjust_targets=True)
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

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
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

cost = T.mean(T.nnet.binary_crossentropy(py_x, Y))
params = [w_h, w_o]
update = sgd(cost, params, lr=0.001) 
# update = RMSprop(cost, params)

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)
compute_H = theano.function(inputs=[X], outputs=h, allow_input_downcast=True)
batch_size=80
for i in range(15):
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train(trX[start:end], trY[start:end])
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

