import numpy as np

from foxhound.models import Network
from foxhound import ops
from foxhound import iterators
from foxhound.transforms import OneHot
from foxhound.theano_utils import floatX

from sklearn.metrics import accuracy_score

from faces import faces

binary_output = True

trX, trY, teX, teY = faces(zscore=True)

def model_perceptron(input_shape):
    model = [
        ops.Input(['x', input_shape])
      , ops.Project(dim=1)
      , ops.Activation('sigmoid')
    ]
    return model

def model_MLP(input_shape):
    model = [
        ops.Input(['x', input_shape])
      , ops.Project(dim=256)
      , ops.Activation('tanh')
      , ops.Project(dim=1)
      , ops.Activation('sigmoid')
      ]
    return model

# Learn and Predict
trXt = lambda x: floatX((np.asarray(x)))
teXt = trXt
if binary_output:
    trYt = lambda y: floatX(y)
else:
    trYt = lambda y: floatX(OneHot(y, 2))
iterator = iterators.Linear(size=100, trXt=trXt, teXt=teXt, trYt=trYt)
model = model_MLP(trX.shape[1])
model = Network(model, iterator=iterator)
 
continue_epochs = True
min_cost_delta = .00001
min_cost = .001
cost0, cost1 = None, None
epoch_count = 0

while continue_epochs:
    epoch_count += 1
    costs = model.fit(trX, trY)
    if cost0 is None:
        cost0 = costs[-1]
    elif cost1 is None:
        cost1 = costs[-1]
    else:
        if ( (cost1 - cost0) <= min_cost_delta ) and (cost1 <= min_cost):
            continue_epochs = False
    # Eval Train/Test Error Every N Epochs
    if epoch_count % 1 == 0:
        if binary_output:
            trYpred = model.predict(trX) > 0.5
            teYpred = model.predict(teX) > 0.5
        else:
            trYpred = np.argmax(model.predict(trX), axis=1)
            teYpred = np.argmax(model.predict(teX), axis=1)
        print trY.shape
        print trYpred.shape
        train_error = 1 - accuracy_score(trY.flatten(), trYpred)
        test_error = 1 - accuracy_score(teY.flatten(), teYpred)
        print "Train Error: ", train_error
        print "Test Error: ", test_error
