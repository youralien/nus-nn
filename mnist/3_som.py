import numpy as np
from load import mnist
# from foxhound.utils.vis import grayscale_grid_vis, unit_scale
from scipy.misc import imsave
from scipy.spatial.distance import cdist

def init_weights(shape):
    return np.random.randn(*shape) * 0.01

def idx1D_to_idx2D(idx, shape):
    n_rows, n_cols = shape

    ith_row = idx / n_cols
    jth_col = idx % n_cols

    return (ith_row, jth_col)

def test_idx1D_to_idx2D():
    assert idx1D_to_idx2D(0, (3,3)) == (0, 0)
    assert idx1D_to_idx2D(1, (3,3)) == (0, 1)
    assert idx1D_to_idx2D(7, (3,3)) == (2, 1)
# test_idx1D_to_idx2D()

def model(X, w_h):
    lattice = (X, w_h)
    return lattice

# trX, teX, trY, teY = mnist(onehot=True)

# if raw_input('remove_classes 3 and 4? (y/n)') == 'y':
#     trY = np.column_stack((trY[:,:3], trY[:,5:]))
#     teY = np.column_stack((teY[:,:3], teY[:,5:]))
#     assert teY.shape[1] == 8
#     assert trY.shape[1] == 8
#     n_classes = 8
# else:
#     assert teY.shape[1] == 10
#     n_classes = 10

X = np.random.randn(1,784)
Y = np.random.randn(1,8)

map_shape = (5, 4)
map_size = map_shape[0] * map_shape[1]
w_h = init_weights((784, map_size))

distances = cdist(X, w_h.T, 'euclidean')
print distances.shape
print distances

winner_idx = np.argmin(distances)
winner_i, winner_j = idx1D_to_idx2D(winner_idx, map_shape)
print "winner: ", winner_i, winner_j

neighbors = []
for count in range(map_size):
    neighbors.append(idx1D_to_idx2D(count, map_shape))
print neighbors
neighbors = np.vstack(neighbors)
print neighbors
print neighbors.shape

winner_idx2D_vector = np.array((winner_i, winner_j)).reshape(1, -1)
map_distances = cdist(winner_idx2D_vector, neighbors)
print map_distances.reshape(map_shape)

def time_varying_sigma(n, sigma_0, tau):
    sigma = sigma_0 * np.exp(-n / float(tau))
    return sigma

def time_varying_neighborhood_function(d, n, sigma_0=1, tau=1):
    """
    d: distance from neighbor
    n: iteration. n=0 is the start of time
    """
    h = np.exp( d**2 / 2.*time_varying_sigma(n, sigma_0, tau) )
    return h

n = 0
hs = np.array(
        [time_varying_neighborhood_function(d, n)
            for d in map_distances]
     )

print "hs: \n", hs.reshape(map_shape)
# py_x = model(X, w_h)
# y_x = T.argmax(py_x, axis=1)

# cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
# params = [w_h, w_o]
# updates = sgd(cost, params)

# train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
# predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

# for i in range(100):
#     for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
#         cost = train(trX[start:end], trY[start:end])
#     print np.mean(np.argmax(teY, axis=1) == predict(teX))

