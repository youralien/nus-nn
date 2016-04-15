function optimal_policy = calculateOptimalPolicy(Q)
% calculateOptimalPolicy: given Q values, calculate an optimal policy, by
% taking the max of the Q values as the optimal action for that state.

% Q: size (n_states, n_actions), with Q values being the worth of taking
% that action at that state

% optimal_policy: size (n_states, 1), with values being the action one
% should take
    [~, optimal_policy] = max(Q, [], 2);
end
