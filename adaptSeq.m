function net = adaptSeq( n, xTrain, tTrain, epochs, learningRate )
%ADAPTSEQ Sequential training of the 1-n-1 MLP using the "adapt" function
% n: number of hidden neurons
% xTrain: the input of training set
% tTrain: the target of training set
% epochs: number of epochs of training
% learningRate: the learning rate of training
%see document:
% http://www.mathworks.com/help/nnet/ug/neural-network-training-concepts.html
% 1. Change the input to cell array form for sequential training
P = num2cell(xTrain, 1);
T = num2cell(tTrain, 1);
% 2. Configure the network
net = fitnet(n); % or feedforwardnet, or newff in old version MATLAB
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;
net.inputWeights{1,1}.learnParam.lr = learningRate;
net.layerWeights{2,1}.learnParam.lr = learningRate;
net.biases{1}.learnParam.lr = learningRate;
net.biases{2}.learnParam.lr = learningRate;
% 3. Train the network in sequential mode.
for ii = 1:epochs
 index = randperm(size(P, 2));
 net = adapt(net, P(:,index), T(:,index));
end
end
