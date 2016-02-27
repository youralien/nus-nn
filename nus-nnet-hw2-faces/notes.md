# batch mode 

- 800 examples to compute the gradient
- SGD with learning rate 0.1
- train error: 0.15
- test error: 0.17
- over epochs, the error does not change. seems like this run was a lucky initialization.

# sequential mode

- 1 example at a time, for an entire epoch
- SGD with learning rate 0.1
- train error: 0.15
- the training examples were shuffled each epoch
- train error: 0.08
- test error: 0.13
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
- DRAMATIC improvements in convergence and test error. Converged in 10 epochs, as opposed to 30 +. Also achieves 0.0% test error?!
- TODO: should check if theres a weird thing in my calculations
