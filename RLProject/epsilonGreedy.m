function a_k = epsilonGreedy(Q, moveValidity, s_k, epsilon)
% Given a estimated worth function for your game board, a current state, and a value
% epsilon which determines how likely you are to explore non-exploitative
% actions, output the action to take at this state, based on the epsilon
% greedy strategy.

% Q: Q matrix, size (n_state, n_action)
% moveValidty: move validity matrix, size (n_state, n_actions). -1 denotes an
%   invalid move, all other values are valid.
% s_k: current state, int
% epsilon: epsilon value for the policy, float between (0 - 1)

% a_k: action to take at the current state, int

validity_for_actions = moveValidity(s_k,:);
worth_for_actions = Q(s_k,:);

valid_actions = find(validity_for_actions >= 0); % FIXME: >=!!

% BIASED: Don't used based on heatmap viz
% for worth values equal, max returns first instance, and thus is biased towards actions
% [~, a_k_exploit] = max(worth_for_actions);

% UNBIASED: If multiple values equal, take random exploitation
max_worth = max(worth_for_actions(valid_actions));
a_k_exploit = valid_actions(find(worth_for_actions(valid_actions) == max_worth)); % get one of 4 indexes (actions) which has should be exploited
if length(a_k_exploit) > 1
    a_k_exploit = randsample(a_k_exploit, 1); % choose a random exploit option if there are multiple
end

if rand() < epsilon % exploration
    
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