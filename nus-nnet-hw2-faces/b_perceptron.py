import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.metrics import accuracy_score
from faces import faces, permute

srng = RandomStreams()

trX, trY, teX, teY = faces(zscore=True, perceptron=True)
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

for epoch in range(100):
    if isinstance(batch_size, int):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            cost = train(trX[start:end], trY[start:end])
    else:
        cost = train(trX, trY)
    if epoch % 1 == 0:
        print "%d,%0.4f,%0.4f" % (epoch, 1-np.mean(np.sign(trY)== predict(trX)), 1-np.mean(np.sign(teY)== predict(teX)))
        trX, trY = permute(trX, trY)
