load('task1.mat')
thresh = 0.05;
decay_type = 1; % this one never worked before...
[Q, ~, time, success_rate] = QLearnManyTrial(reward, 0.9, decay_type, thresh, 0);
optimal_policy = calculateOptimalPolicy(Q);
totalReward = walkOptimalPolicy(optimal_policy, reward);