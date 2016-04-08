function ys = QLearner
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


    function a_k = epsilonGreedy(reward, s_k, epsilon)
        if rand() < epsilon % exploration
            valid_actions = find(reward(s_k,:) > 0);
            a_k = randsample(valid_actions, 1);
        else % exploitation
            reward_for_actions = reward(s_k,:);
            [~, a_k] = max(reward_for_actions);
        end
    end

    function s_k1 = transition(s_k, a_k, n_states, shp)
        if nargin < 3 % default to 100 states, 10x10 grid
            n_states = 100;
            shp = [10 10];
        end
        action_movements = [
            -1, 0; % up
            0, 1; % right
            1, 0; % down
            0,-1;]; % left
        s_k_sub = ind2sub(n_states,s_k); % current state i,j
        s_k1_sub = s_k_sub + action_movements(a_k,:); % next state i,j
        s_k1 = sub2ind(shp, s_k1_sub(1), s_k1_sub(2)); % next state
    end

    function mult = decay(k, type)
        if type == 1
            mult = 1 / k;
        elseif type == 2
            mult = 100 / (100 + k);
        elseif type == 3
            mult = (1 + log(k)) / k;
        else
            mult = (1 + 5*log(k)) / k;
        end
        if mult > 1
            mult = 1;
        end
    end

s_k = 1
a_k = epsilonGreedy(task1.reward, s_k, 0.2)
s_k1 = transition(s_k, a_k)

iter = 300;
ys = zeros(iter,4);
for type=1:4
    for k=1:iter
        ys(k,type) = decay(k,type);
    end
end

end
