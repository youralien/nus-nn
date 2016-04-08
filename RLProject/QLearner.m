function QLearner
% Parameters
% ----------
% epsilon: float (0 - 1)
%   determines epsilon greedy strategy
% alpha: float (0 - 1)
%   determines the learning rate for the Q update
% gamma: float (0 - 1)
%   indicates the discount factor, a design parameter of the algo
% Q: array-like, shape (100, 4)
%   all the previous Q values of the state-action board. Each row specifies
%   the particular state, and each column indicates actions 1 - 4 for that
%   state.
% N: array-like, shape (100, 4)
%   number of times we've traveled through particular state action pairs.
% reward: array-like, shape (100, 4)
%   reward matrix specifying what type of reward we obtain by taking some
%   action at some state
% Returns
% -------
% Q: all the updates Q values of the state-action board
% N: all the updated N values of the state-action board
task1 = load('task1.mat'); % has the reward variable
Q = zeros(100,4);
N = zeros(100,4);

    
    function a_k = epsilonGreedy(reward, state, epsilon)
        if rand() < epsilon % exploration
            valid_actions = find(reward(state,:) > 0);
            a_k = randsample(valid_actions, 1);
        else % exploitation
            reward_for_actions = reward(state,:)
            [~, a_k] = max(reward_for_actions);
        end
    end

a_k = epsilonGreedy(task1.reward, 1, 0.2)

end