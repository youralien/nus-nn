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

def contrast_normalize(matrix):
    for row in range(matrix.shape[0]):
        mu = np.mean(matrix[row])
        sigma = np.std(matrix[row])
        matrix[row] = (matrix[row] - mu) / sigma
    return matrix

def faces(zscore=False, onehot=False, adjust_targets=False, perceptron=False, contrast=False):

    tr_features = pd.read_csv('TrainFeats.csv', header=None)
    tr_target = pd.read_csv('TrainLabels.csv', header=None)
    te_features = pd.read_csv('TestFeats.csv', header=None)
    te_target = pd.read_csv('TestLabels.csv', header=None)

    def helper(X, Y, zscore, adjust_targets, perceptron, constrast):
        X = np.asarray(X, dtype='float32')
        Y = np.asarray(Y, dtype='float32')

        if contrast:
            print "faces.py: contrast normalization"
            X = contrast_normalize(X)
        
        if zscore:
            print "faces:py: make features zero mean and unit variance"
            X = z_score(X)

        if adjust_targets:
            print "faces.py: adjusting targets to be in the sigmoid range"
            Y[np.where(Y == 1)] = 0.8
            Y[np.where(Y == 0)] = 0.2

        if perceptron:
            print "faces.py: targets are [-1, 1]"
            Y[np.where(Y == 0)] = -1

        return X, Y
   
    # ensure that zscore is calculated in relation to all data, train + test
    X = np.vstack((tr_features.values, te_features.values))
    Y = np.vstack((tr_target.values, te_target.values))
    X, Y = helper(X, Y, zscore, adjust_targets, perceptron, contrast)
    cutoff = tr_features.values.shape[0]
    trX = X[:cutoff]
    trY = Y[:cutoff]
    teX = X[cutoff:]
    teY = Y[cutoff:]

    if onehot:
        trY = one_hot(trY, 2)
        teY = one_hot(teY, 2)
    return trX, trY,teX, teY

if __name__ == "__main__":
    trX, trY, teX, teY = faces(zscore=True, adjust_targets=True)
    print trX.shape, trY.shape, teX.shape, teY.shape
    xcol = trX[:,np.random.randint(trX.shape[1])]
    print "X min: {}, X max: {}, X mean: {}, X variance: {}".format(xcol.min(), xcol.max(), xcol.mean(), xcol.std()**2)
    print "Y min: {}, Y max: {}".format(trY.min(), trY.max())
