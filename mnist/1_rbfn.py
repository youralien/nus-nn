import numpy as np

func = lambda x: 1.2*np.sin(np.pi*x) - np.cos(2.4*np.pi*x)
noisy_func = lambda x: func(x) + 0.3*np.random.randn()

trX = np.arange(-1,1,0.05)
teX = np.arange(-1,1,0.01)
trY = noisy_func(trX) # observed training data is noisy
teY = func(teX)       # actual test data is the real function
#TODO: plot y values
n_tr = trX.shape[0]
n_te = teX.shape[0]

rbf = lambda r, std: np.exp(-r**2 / (2.*std**2) )
# TODO: test rbf
def test_rbf():
    rs = [0, 0.05, 0.5]
    outs = [rbf(r, 0.1) for r in rs]
    for i in range(len(outs)-1):
        assert outs[i] > outs[i+1] # monotonically decreasing w/ increasing r
test_rbf()

class ExtactInterpolationRBFNetwork():
    """Exact Interpolation is the special case where

    interpM.shape = (N, N)
    where N is number of training examples

    and the centers of the RBF activations are the training data
    mu_j = x_train_j
    """
    def __init__(self, std=0.1):
        self.std = std

    def fit(self, trX, trY):

        n_tr = trY.shape[0]
        interpM = np.zeros((n_tr, n_tr), dtype='float32')
        for i in xrange(n_tr):
            for j in xrange(n_tr):
                interpM[i,j] = rbf(np.linalg.norm(trX[i] - trX[j]), self.std)
        # TODO: interp matrix should have diag of 1's
        # TODO: interp matrix should also have interesting vals outside of diagonals
        self.w = np.dot(trY,np.linalg.inv(interpM))
        self.mu = trX # mu is the training examples
        # TODO: w should be the same shape as trY
        # TODO: interpM * w should yield good accuracies
        assert self.w.shape == trY.shape

    def predict(self, teX):
        act = np.zeros((teX.shape[0], self.w.shape[0]), dtype='float32')
        # iterate over each test example to predict
        for i in range(teX.shape[0]):
            # process through the activation function for each hidden neuron
            for j in range(self.w.shape[0]):
                act[i,j] = rbf(np.linalg.norm(teX[i] - model.mu[j]), self.std)
        teXpred = np.dot(act, model.w)
        return teXpred


model = ExtactInterpolationRBFNetwork()
model.fit(trX, trY)
teXpred = model.predict(teX)
error = np.mean(np.abs(teXpred - teY))

with open('exactrbfn_results.txt', 'aw') as f:
    f.write("{}\n".format(error))
