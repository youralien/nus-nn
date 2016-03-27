% svm_main.m
% assume that the variable "data", "label" and "evaldata" present
% desired output: "evallabel"

% Asserting Shapes are correct
n_train = 285;
n_eval = 100;
n_features = 30;
if size(data) ~= [n_features, n_train]
    disp('Fixing shape of train data');
    data = data';
end
if size(label) ~= [n_train, 1]
    disp('Fixing shape of train label');
    label = label';
end 
if size(evaldata) ~= [n_features, n_eval]
    disp('Fixing shape of eval data');
    evaldata = evaldata';
end

svm = SVMEstimator(2.1,'poly',3,1e-7);
svm.fit(data, label);
evallabel = svm.predict(evaldata);