function mult = decay(k, type)
% specifies the decay function. In the context of the basic Q learning
% algorithm, we decay epsilon and alpha based on the number of transitions.

% k: transition number, int
% type: the type of function, int between 1 - 4

% mult: multiplier between (0 - 1) specifying how much to decay the initial
% value

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
