import numpy as np
from load import mnist
# from foxhound.utils.vis import grayscale_grid_vis, unit_scale
from scipy.misc import imsave
from scipy.spatial.distance import cdist

def init_weights(shape):
    return np.random.randn(*shape) * 0.01

def idx1D_to_idx2D(idx, shape):
    n_rows, n_cols = shape

    ith_row = idx / n_cols # integer division
    jth_col = idx % n_cols

    return (ith_row, jth_col)

def test_idx1D_to_idx2D():
    assert idx1D_to_idx2D(0, (3,3)) == (0, 0)
    assert idx1D_to_idx2D(1, (3,3)) == (0, 1)
    assert idx1D_to_idx2D(7, (3,3)) == (2, 1)
# test_idx1D_to_idx2D()

def time_varying_sigma(n, sigma_0, tau):
    sigma = sigma_0 * np.exp(-n / float(tau))
    return sigma

def time_varying_neighborhood_function(d, n, sigma_0, tau):
    """
    d: distance from neighbor
    n: iteration. n=0 is the start of time
    """
    h = np.exp( d**2 / 2.*time_varying_sigma(n, sigma_0, tau) )
    return h

def learningRate(n, lr_0, n_iter_first_phase, lr_min=0.01):
    lr = lr_0 * np.exp( -n / float(n_iter_first_phase) )

    if lr < lr_min:
        return lr_min
    else:
        return lr

def init_neighborhood_size(map_shape):
    m, n = map_shape
    sigma_0 = np.sqrt(m**2 + n**2) / 2.
    return sigma_0

def init_timeconstant(n_iter_first_phase, sigma_0):
    return float(n_iter_first_phase) / np.log(sigma_0)

trX, teX, trY, teY = mnist(onehot=True)

if raw_input('remove_classes 3 and 4? (y/n)') == 'y':
    trY = np.column_stack((trY[:,:3], trY[:,5:]))
    teY = np.column_stack((teY[:,:3], teY[:,5:]))
    assert teY.shape[1] == 8
    assert trY.shape[1] == 8
    n_classes = 8
else:
    assert teY.shape[1] == 10
    n_classes = 10

# X = np.random.randn(1,784)
# Y = np.random.randn(1,8)

map_shape = (5, 4)
map_size = map_shape[0] * map_shape[1]
w = init_weights((784, map_size))

n_iter_first_phase = 1000;
sigma_0 = init_neighborhood_size(map_shape)
tau = init_timeconstant(n_iter_first_phase, sigma_0)
lr_0 = 0.1

for n in range(n_iter_first_phase):
    # MNIST
    x = trX[n % trX.shape[0]].reshape(1, -1)
    distances = cdist(x, w.T, 'euclidean')
    print distances.shape
    print "distances: \n", distances

    winner_idx = np.argmin(distances)
    winner_i, winner_j = idx1D_to_idx2D(winner_idx, map_shape)
    print "winner: ", winner_i, winner_j

    neighbors = []
    for count in range(map_size):
        neighbors.append(idx1D_to_idx2D(count, map_shape))
    neighbors = np.vstack(neighbors)
    print "neighbors: \n", neighbors

    winner_idx2D_vector = np.array((winner_i, winner_j)).reshape(1, -1)
    map_distances = cdist(winner_idx2D_vector, neighbors)
    print "map_distances: \n", map_distances.reshape(map_shape)

    lr = learningRate(n, lr_0, n_iter_first_phase)
    hs = np.array(
            [time_varying_neighborhood_function(d, n, sigma_0, tau=tau)
                for d in map_distances]
         )
    
    # -- weight update
    # w_new = w + lr*hs*(np.tile(x, (map_size,1)).T - w) # vectorized
    # readable for loop
    for neuron_idx in range(map_size):
        w[:,neuron_idx] = w[:,neuron_idx] + lr*hs[:,neuron_idx]*(x - w[:,neuron_idx])  


# py_x = model(X, w)
# y_x = T.argmax(py_x, axis=1)

# cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
# params = [w, w_o]
# updates = sgd(cost, params)

# train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
# predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

# for i in range(100):
#     for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
#         cost = train(trX[start:end], trY[start:end])
#     print np.mean(np.argmax(teY, axis=1) == predict(teX))

