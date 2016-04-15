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
[Q, N] = QLearnManyTrial(discount, decay_type, thresh, do_plot);

end
