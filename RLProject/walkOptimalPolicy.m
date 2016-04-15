function totalReward = walkOptimalPolicy( optimal_policy, reward )
%walkOptimalPolicy follow the optimal policy (given by max Q values) to try
%to see what the total return is.
% This code was modified to return the reward of the optimal policy, but
% originated by code called plot_grid.m, by Rob Romijnders

%   optimal_policy: array-like, size (n_states,1) values giving actions
%   reward: size (n_states, n_actions), values are the reward taking action
%   at a given state

%   rewardTotal: total reward by following optimal policy. if no optimal policy,
%   reward returned is -1

state = 1;
n = 0;
totalReward = 0;
while( state ~= 100 && n < 100)
    % Define the next step and the symbol that indicates the direction
    action = optimal_policy(state);
    totalReward = totalReward + reward(state, action);
    state = transition(state, action);
    n = n + 1;
end
if state ~= 100 % never reached the goal
    totalReward = -1;
end
end

