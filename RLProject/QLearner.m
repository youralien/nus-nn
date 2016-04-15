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

    function a_k = epsilonGreedy(reward, s_k, epsilon)
        reward_for_actions = reward(s_k,:);
        % for reward values equal, max returns first instance, and thus is biased towards actions
        [~, a_k_exploit] = max(reward_for_actions);

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
        [s_k_i, s_k_j] = ind2sub(shp,s_k); % current state i,j
        s_k1_sub = [s_k_i, s_k_j] + action_movements(a_k,:); % next state i,j
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

    function [Q, N] = QLearnOneTrial(Q, N, discount, decay_type)
        iterations = 10000; % some big number, almost like a while loop
        s = ones(iterations,1);
        a = ones(iterations,1);

        for k=1:iterations
            alpha = decay(k, decay_type); % type of decay?
            epsilon = alpha; % epsilon and alpha the same
            a(k) = epsilonGreedy(task1.reward, s(k), epsilon);
%             disp([s(k) a(k)]);
            s(k+1) = transition(s(k), a(k));
            N(s(k),a(k)) = N(s(k),a(k)) + 1;
            Q(s(k),a(k)) = Q(s(k),a(k)) + alpha*(task1.reward(s(k),a(k)) + discount*max(Q(s(k+1),:)) - Q(s(k),a(k)));
            if s(k+1) == 100 || alpha < 0.005 % stop condition for trial
                disp(['final state: ' num2str(s(k+1)) ', alpha: ' num2str(alpha)]);
                break
            end
        end
    end

    function [Q, N] = QLearnManyTrial(discount, decay_type, convergence_thresh, plotting)
        Q = zeros(100,4);
        N = zeros(100,4);
        diffQ = ones(3000,1);
        for trial=1:3000
            [Qnew, N] = QLearnOneTrial(Q, N, discount, decay_type);
            diffQ(trial) = mean((Qnew(:) - Q(:)) .^ 2);
            if trial > 1
                % make diffQ a moving average of previous values
                diffQ(trial) = 0.05*diffQ(trial) + 0.95*diffQ(trial-1);
                if diffQ(trial) < convergence_thresh
                    break
                end
                if plotting && mod(trial, 5) == 0
                    plot(diffQ(1:trial)); ylabel('||Qnew - Q||'); xlabel('Trials');
                    pause(0.1);
                end
            end
            Q = Qnew;
        end
    end

discount = 0.9;
decay_type = 2;
thresh = 0.05;
do_plot = true;
[Q, N] = QLearnManyTrial(discount, decay_type, thresh, do_plot)

end
