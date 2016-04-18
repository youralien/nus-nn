function mult = decay(k, type)
% specifies the decay function. In the context of the basic Q learning
% algorithm, we decay epsilon and alpha based on the number of transitions.

% k: transition number, int
% type: the type of function, int between 1 - 4

% mult: multiplier between (0 - 1) specifying how much to decay the initial
% value
if type == 0
    mult = 0.5; % constant rate
elseif type == 1
    mult = 1 / k;
elseif type == 2
    mult = 100 / (100 + k);
elseif type == 3
    mult = (1 + log(k)) / k;
elseif type == 4
    mult = (1 + 5*log(k)) / k;
else
    deltaxs = [100, 300, 600, 900];
    deltax = deltaxs(type - 4);
    % linear decay to 300?  300 came from the xlim of my original decay
    % graph.
    deltay = 0.9; % 1 - 0.1
    if k > deltax
        mult = 0.1;
    else
        mult = -(deltay/deltax)*k + 1; % y = mx + b
    end
end
if mult > 1
    mult = 1;
end
end
