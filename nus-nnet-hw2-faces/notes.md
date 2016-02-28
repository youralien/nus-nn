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
