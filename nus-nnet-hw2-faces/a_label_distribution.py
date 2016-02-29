import pandas as pd
import numpy as np

dataset = 'Train'
# dataset = 'Test'
target = pd.read_csv(dataset + 'Labels.csv', header=None)
Y = target.values
rows = Y.shape[0] 

neg = np.sum(Y == 0) / float(rows) 
pos = np.sum(Y == 1) / float(rows)

print "Task-ID1 - Gender Classification"
print "Negative Examples: %f" % neg
print "Postive  Examples: %f" % pos
