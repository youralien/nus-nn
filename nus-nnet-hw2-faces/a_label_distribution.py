import pandas as pd
import numpy as np

target = pd.read_csv('unnormalized_data_faces_target.csv')
Y = target.values
rows = Y.shape[0] 

neg = np.sum(Y == 0) / float(rows) 
pos = np.sum(Y == 1) / float(rows)

print "Task-ID1 - Gender Classification"
print "Negative Examples: %f" % neg
print "Postive  Examples: %f" % pos
