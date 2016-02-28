# batch mode 

- SGD with learning rate 0.05
- 100 epochs
- train error: 0.04
- test error: 0.05

# sequential mode

- 1 example at a time, for an entire epoch
- SGD with learning rate 0.0005
- 100 epochs
- the training examples were shuffled each epoch
- train error: 0.0
- test error: 0.08
- over epochs, the error does fluctate, even becoming worse at times (stochastic jumping in and out of minima)

# mini batch mode

- 80 examples for a gradient update, doing that for an entire epoch
- SGD with learning rate 0.1
- train error: 0.15
- the training examples were shuffled each epoch
- train error: 0.0
- test error: 0.07
- these favorable metrics were achieved more quickly than the other methods, approaching these metrics within a handlful of epochs

# adjusting the target values to be within the sigmoid range
- TODO: observe effects now that I fixed the bug

-------------------------------------------------------------------------------------------------------------------------

# batch mode results: 

zscore=True, adjust_targets=True, SGD lr=0.05, uniform weight initialization [-0.02, 0.02], no dropout
note on architecture: 10202 -> 128 -> 1, {tanh, sigmoid}. Rank of final H is ~120 on multiple tries (close enough to 128)

train error, test error
0.0725     , 0.1256
0.0663     , 0.1407
0.1012     , 0.1106
0.0625     , 0.1106
0.0900     , 0.1307

without adjusting the target values, for personal interest:
train error, test error
0.0900     , 0.1256
0.0663     , 0.0955
0.0550     , 0.0804
0.0713     , 0.0854
0.0513     , 0.1658

While making the targets [0.2, 0.8] gives a more consistent test error, without adjustment may give on average a better test score.

# sequential mode results:
zscore=True, adjust_targets=True, SGD lr=0.0005, uniform weight initialization [-0.02, 0.02], no dropout
note on architecture: 10202 -> 128 -> 1, {tanh, sigmoid}. Rank of final H is 121 (close enough to 128)

train error, test error
0.0000     , 0.0653
0.0000     , 0.0452
0.0000     , 0.0854
0.0000     , 0.0754
0.0000     , 0.0704

one thing really interesting is that sequential mode is able to make progress after the first epoch.  Here is the start and end values of the learning curve for sequential mode:

epoch, train , test
0    , 0.1175, 0.1206
90   , 0.0000, 0.0653

even with a good initialization, it still manages to improve and completely reduce the error in the training set, while improving test set performance.

epoch, train , test
0    , 0.1362, 0.0553
90   , 0.0000, 0.0452

compare this to batch mode learning, where updates are slow and often we are stuck with local minima
epoch, train , test
0    , 0.1188, 0.1407
90   , 0.0663, 0.1156

with a good initialization, the results are desirable as we obtain good error measures for both train and test (no overfitting happens, coincidentally).  However, the test error is still higher than the two learning procedures using sequential mode.

epoch, train , test
0    , 0.1288, 0.0955
90   , 0.0763, 0.0704

