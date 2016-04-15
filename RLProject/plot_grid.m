function [] = plot_grid(optimal_policy)

%
% This function enables to plot a grid with the optimal policy found by
% Q-learning algorithm for each run.
%   - optimal_policy : vector of size 100. The nth value corresponds to the
%   action to take for the nth step.
% Returns a figure
%

% Define a figure
figure
grid on
axis([0 10 0 10])

state = 1;
n = 0;
while( state ~= 100 && n < 100)
    % Convert the state into coordinates x and y
    [y,x] = ind2sub([10,10],state);
    % Define the next step and the symbol that indicates the direction
    if optimal_policy(state) == 1
        text(x - 0.5, 10 - y + 0.5, '^')
        state = state - 1;
    elseif optimal_policy(state) == 2
        text(x - 0.5, 10 - y + 0.5, '>')
        state = state + 10;
    elseif optimal_policy(state) == 3
        text(x - 0.5, 10 - y + 0.5, 'v')
        state = state + 1;
    else
        text(x - 0.5, 10 - y + 0.5, '<')
        state = state - 10;
    end
    
    n = n + 1;
    
end

% Indicate the the first and last state 
text(0.35,9.25, '*', 'color', 'g', 'FontSize', 30);
text(9.35,0.25, '*','color', 'r', 'FontSize', 30);


