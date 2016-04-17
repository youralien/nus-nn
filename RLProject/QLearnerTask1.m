function results_table = QLearnerTask1()
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

thresh = 0.5;
do_plot = false;
runs = 10;
table_goal_reached_runs = zeros(4,2);
table_execution_time = zeros(4,2);
discounts = [0.5, 0.9];
for i=1:length(discounts)
    discount = discounts(i);
    for decay_type=[1 2 3 4] % epsilon and alpha decay the same
        Qs = zeros(100, 4, runs);
        success_rates = zeros(runs, 1);
        optimal_policy = zeros(100, runs);
        totalReward = zeros(runs, 1);
        times = zeros(runs,1);
        parfor run=1:runs
            [Qs(:,:,run), ~, times(run), success_rates(run)] = QLearnManyTrial(task1.reward, discount, decay_type, decay_type, thresh, do_plot);
            optimal_policy(:,run) = calculateOptimalPolicy(Qs(:,:,run));
            totalReward(run) = walkOptimalPolicy(optimal_policy(:,run), task1.reward);
        end
        % save the outputs required for the Q learning param values and
        % performance table
        goalReachedRuns = find(totalReward > 0);
        table_goal_reached_runs(decay_type,i) = length(goalReachedRuns);
        table_execution_time(decay_type,i) = mean(times(goalReachedRuns));
        % final output should be an optimal policy (if goal state reached
        % in the 10 runs) and the reward associated with this optimal
        % policy. optimal policy is a state vector, and also should be
        % visualized in the 10 x 10 grid
        if length(goalReachedRuns) > 0
            fn = ['policy_discount' num2str(discount) '_decay' int2str(decay_type) '.mat'];
            save(fn, 'optimal_policy', 'totalReward');
        end
        disp(['DONE: ' num2str(discount) int2str(decay_type)]);
    end
end
results_table = [table_goal_reached_runs, table_execution_time];
end
