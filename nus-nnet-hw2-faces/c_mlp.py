import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.metrics import accuracy_score
from faces import faces, permute

from foxhound.inits import Orthogonal

srng = RandomStreams()

trX, trY, teX, teY = faces(zscore=True, onehot=False)
print trX.shape
print trY.shape
input_dim = trX.shape[1]


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    orth_init = Orthogonal()
    return orth_init(shape)
    # return theano.shared(floatX(np.random.randn(*shape) * 0.02))

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
    return pyx

X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((input_dim, 50))
w_o = init_weights((50, 1))

py_x = model(X, w_h, w_o, 0.5, 0.5)
y_pred = model(X, w_h, w_o, 0., 0.) > 0.5

cost = T.mean(T.nnet.binary_crossentropy(py_x, Y))
params = [w_h, w_o]
# update = sgd(cost, params, lr=0.1) 
update = RMSprop(cost, params)

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)
batch_size=100
for i in range(100):
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train(trX[start:end], trY[start:end])
    print i, np.mean(trY == predict(trX)), np.mean(teY == predict(teX))
    trX, trY = permute(trX, trY) 
    # print i, np.mean(np.argmax(teY, axis=1) == predict(teX))
