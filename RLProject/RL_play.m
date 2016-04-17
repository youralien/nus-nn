load('task1.mat')
thresh = 0.5;
a_decay_type = 2; % this one never worked before...
e_decay_type = 2;
gamma = 0.9;
[Q, N, time, success_rate] = QLearnManyTrial(reward, gamma, a_decay_type, e_decay_type, thresh, 1);
optimal_policy = calculateOptimalPolicy(Q);
totalReward = walkOptimalPolicy(optimal_policy, reward);
heatmap(N);