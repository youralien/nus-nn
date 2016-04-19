% RL_main - perform Qlearning with the best parameters, on an unknown
% reweard function
% 'reward' variable will already exist, loaded from qeval.mat
thresh = 1;
do_plot = false;
discount = 0.9;
a_decay_type = 0;
e_decay_type = 2;
[Q, N, time, success_rate] = QLearnManyTrial(reward, discount, a_decay_type, e_decay_type, thresh, do_plot);
qevalstates = calculateOptimalPolicy(Q);
totalReward = walkOptimalPolicy(qevalstates, reward);