import numpy as np
from scipy.spatial.distance import pdist

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

class SpyderObject(object):
    """ An object that makes object attributes explorable in Spyder """
    def __init__(self):
        pass

    def exploreClassAttributes(self):
        for variable_name, variable_value in self.__dict__.iteritems():
            locals()["self_" + variable_name] = variable_value
        # delete so there are no duplicate variables in the variable explorer
        del(variable_name)
        del(variable_value)
        return # Place Spyder Breakpoint on this Line!

class ExtactInterpolationRBFNetwork(SpyderObject):
    """Exact Interpolation is the special case where

    interpM.shape = (N, N)
    where N is number of training examples

    the centers of the RBF activations are the training data
    mu_j = x_train_j

    the standard deviations are at a fixed width (default 0.1)
    """
    def __init__(self, std=0.1):
        super(ExtactInterpolationRBFNetwork, self).__init__()
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
                act[i,j] = rbf(np.linalg.norm(teX[i] - self.mu[j]), self.std)
        teXpred = np.dot(act, self.w)
        return teXpred

class RandomFixedCentersRBFNetwork(SpyderObject):
    """Random Fixed Centers is the case where

    interpM.shape = (N, H)
    where H is number of hidden nodes chosen as a parameter

    the centers of the RBF activations are H samples from the train data
    mu_j = x_train_j

    the standard deviations are defined by
    sigma_j = d_max / sqrt(2H)
    where d_max is maximum distance between chosen centers
    and H the number of hidden nodes
    """
    def __init__(self, n_hidden=15):
        super(RandomFixedCentersRBFNetwork, self).__init__()
        self.n_hidden = 15 # H parameter

    def fit(self, trX, trY):
        # select random centers from training data
        self.mu = trX[np.random.randint(0, trX.shape[0], self.n_hidden)]

        # precompute params
        if len(self.mu.shape) == 1: # 1D data cant be handled by pdist
            self.d_max = np.max(pdist(self.mu.reshape(-1, 1)))
        else: # 2D data can already be handled by pdist
            self.d_max = np.max(pdist(self.mu))
        self.std = self.d_max / ( np.sqrt(2.*self.n_hidden) )

        # construct interpolation matrix
        n_tr = trY.shape[0]
        interpM = np.zeros((n_tr, self.n_hidden), dtype='float32')
        for i in xrange(n_tr):
            for j in xrange(self.n_hidden):
                interpM[i,j] = rbf(np.linalg.norm(trX[i] - self.mu[j]), self.std)
        if n_tr == self.n_hidden: # interpM is square
            self.w = np.dot(trY,np.linalg.pinv(interpM).T)
        else: # we find the minimum using the same as linear least squares
            # TODO: sometimes interpM.T * interpM is singular, others not
            # FIXME: currently using pinv, not sure if its satisfactory
            self.w = np.dot(
                  np.dot(
                        np.linalg.pinv(np.dot(interpM.T, interpM))
                      , interpM.T)
                , trY)

    def predict(self, teX):
        act = np.zeros((teX.shape[0], self.w.shape[0]), dtype='float32')
        # iterate over each test example to predict
        for i in range(teX.shape[0]):
            # process through the activation function for each hidden neuron
            for j in range(self.w.shape[0]):
                act[i,j] = rbf(np.linalg.norm(teX[i] - self.mu[j]), self.std)
        teXpred = np.dot(act, self.w)
        return teXpred


def question1a():
    model = ExtactInterpolationRBFNetwork()
    model.fit(trX, trY)
    teXpred = model.predict(teX)
    error = np.mean(np.abs(teXpred - teY))

    with open('ExactInterpRBFN.txt', 'aw') as f:
        f.write("{}\n".format(error))


def question1b():
    model = RandomFixedCentersRBFNetwork()
    model.fit(trX, trY)
    teXpred = model.predict(teX)
    # FIXME: errors can be REALLY bad (one was 554 MAE, vs expected 1)
    error = np.mean(np.abs(teXpred - teY))

    with open('RandomFixedCentersRBFN.txt', 'aw') as f:
        f.write("{}\n".format(error))

def calculate_mean(path):
    f = open(path, 'r')
    foo = f.readlines()
    mean = np.mean([float(line.strip('\n')) for line in foo])
    print mean
    return mean

if __name__ == "__main__":
#    question1a()
    question1b()