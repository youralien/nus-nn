import numpy as np
import sys
from load import mnist
from scipy.spatial.distance import cdist

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
    h = np.exp( d**2 / (2.*time_varying_sigma(n, sigma_0, tau)**2) )
    return h

def learningRate(n, lr_0, n_epochs_organizing_phase, lr_min=0.01):
    lr = lr_0 * np.exp( -n / float(n_epochs_organizing_phase) )

    if lr < lr_min:
        return lr_min
    else:
        return lr

def init_neighborhood_size(map_shape):
    m, n = map_shape
    sigma_0 = np.sqrt(m**2 + n**2) / 2.
    return sigma_0

def init_timeconstant(n_epochs_organizing_phase, sigma_0):
    return float(n_epochs_organizing_phase) / np.log(sigma_0)

trX, teX, trY, teY = mnist(ntrain=2000, ntest=1000, onehot=True)
xmin_val = trX[0].min()
xmax_val = trX[0].max()

if raw_input('remove_classes 3 and 4? (y/n)') == 'y':
    trY = np.column_stack((trY[:,:3], trY[:,5:]))
    teY = np.column_stack((teY[:,:3], teY[:,5:]))
    assert teY.shape[1] == 8
    assert trY.shape[1] == 8
    n_classes = 8
else:
    assert teY.shape[1] == 10
    n_classes = 10

map_shape = (5, 4)
map_size = map_shape[0] * map_shape[1]
w = np.random.uniform(xmin_val, xmax_val, (784, map_size))

n_epochs_organizing_phase = 1000;
sigma_0 = init_neighborhood_size(map_shape)
tau = init_timeconstant(n_epochs_organizing_phase, sigma_0)
lr_0 = 0.1

# organizing phase
verbose = False
for n in range(n_epochs_organizing_phase):
    # MNIST
    n_examples = trX.shape[0]
    for sequential_learning_idx in range(n_examples):
        sys.stdout.write("\rExamples Seen: %d" % sequential_learning_idx)
        x = trX[sequential_learning_idx].reshape(1, -1)
        distances = cdist(x, w.T, 'euclidean')
        winner_idx = np.argmin(distances)
        winner_i, winner_j = idx1D_to_idx2D(winner_idx, map_shape)

        neighbors = []
        for count in range(map_size):
            neighbors.append(idx1D_to_idx2D(count, map_shape))
        neighbors = np.vstack(neighbors)

        winner_idx2D_vector = np.array((winner_i, winner_j)).reshape(1, -1)
        map_distances = cdist(winner_idx2D_vector, neighbors)

        lr = learningRate(n, lr_0, n_epochs_organizing_phase)
        hs = np.array(
                [time_varying_neighborhood_function(d, n, sigma_0, tau=tau)
                    for d in map_distances]
             )
        
        # -- weight update
        # w_new = w + lr*hs*(np.tile(x, (map_size,1)).T - w) # vectorized
        # readable for loop
        for neuron_idx in range(map_size):
            w[:,neuron_idx] = w[:,neuron_idx] + lr*hs[:,neuron_idx]*(x - w[:,neuron_idx])  
        if verbose:
            print "distances: \n", distances
            print "winner: ", winner_i, winner_j
            print "neighbors: \n", neighbors
            print "map_distances: \n", map_distances.reshape(map_shape)
        

        sys.stdout.flush()
    raw_input("continue epoch %d?" % (n+1))