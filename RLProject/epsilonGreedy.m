function a_k = epsilonGreedy(reward, s_k, epsilon)
% Given a reward function for your game board, a current state, and a value
% epsilon which determines how likely you are to explore non-exploitative
% actions, output the action to take at this state, based on the epsilon
% greedy strategy.

% reward: reward matrix, size (n_state, n_actions)
% s_k: current state, int
% epsilon: epsilon value for the policy, float between (0 - 1)

% a_k: action to take at the current state, int

reward_for_actions = reward(s_k,:);

% BIASED: Don't used based on heatmap viz
% for reward values equal, max returns first instance, and thus is biased towards actions
% [~, a_k_exploit] = max(reward_for_actions);

% UNBIASED: If multiple values equal, take random exploitation
max_reward = max(reward_for_actions);
a_k_exploit = find(reward_for_actions == max_reward);
if length(a_k_exploit) > 1
    a_k_exploit = randsample(a_k_exploit, 1); % choose a random exploit option if there are multiple
end

if rand() < epsilon % exploration
    valid_actions = find(reward_for_actions > 0);
    % explore choices we would not normally exploit
    while true
        a_k = randsample(valid_actions, 1); % choose one from the valid_actions
        if a_k ~= a_k_exploit
            break
        end
    end
else % exploitation
    a_k = a_k_exploit;
end
end