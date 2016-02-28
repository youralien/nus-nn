import pandas as pd
import numpy as np

def permute(X, Y):
    matrix = np.column_stack((X,Y))
    np.random.shuffle(matrix)
    X = matrix[:,:-1]
    Y = matrix[:,-1].reshape(-1,1)
    return X,Y
    
def traintestset(X, Y, test_size=0.2):
    X, Y = permute(X, Y)
    rows = Y.shape[0]
    cutoff = int(rows*test_size)
    teX = X[:cutoff]
    teY = Y[:cutoff]
    trX = X[cutoff:]
    trY = Y[cutoff:]
    return trX, trY, teX, teY

def one_hot(x,n):
    if type(x) == list:
            x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def z_score(matrix):
    cols = matrix.shape[1] 
    for col in range(cols):
        mu = np.mean(matrix[:,col])
        sigma = np.std(matrix[:,col])
        matrix[:,col] = (matrix[:,col] - mu) / sigma
    return matrix

def faces(zscore=False, onehot=False, adjust_targets=False):

    features = pd.read_csv('unnormalized_data_faces.csv')
    target = pd.read_csv('unnormalized_data_faces_target.csv')

    X = features.values
    Y = target.values

    if zscore:
        print "faces:py: make features zero mean and unit variance"
        X = z_score(X)

    if adjust_targets:
        print "faces.py: adjusting targets to be in the sigmoid range"
        Y = np.asarray(Y, dtype='float32')
        Y[np.where(Y == 1)] = 0.8
        Y[np.where(Y == 0)] = 0.2

    trX, trY, teX, teY = traintestset(X, Y)

    if onehot:
        trY = one_hot(trY, 2)
        teY = one_hot(teY, 2)
    return trX, trY,teX, teY

if __name__ == "__main__":
    trX, trY, teX, teY = faces(zscore=True, adjust_targets=True)
    xcol = trX[:,np.random.randint(trX.shape[1])]
    print "X min: {}, X max: {}".format(xcol.min(), xcol.max())
    print "Y min: {}, Y max: {}".format(trY.min(), trY.max())
