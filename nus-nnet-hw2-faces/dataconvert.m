split = 'Train';
directory = [split,'Images'];
[status, list] = system( ['dir ', directory, '\*.att ', '/B /S' ] );
result = textscan( list, '%s', 'delimiter', '\n' );
labelList = result{1};

[status, list] = system( ['dir ', directory, '\*.jpg ', '/B /S' ] );
result = textscan( list, '%s', 'delimiter', '\n' );
imageList = result{1};

side_length = 101;
shape = size(labelList);
n_examples = shape(1);
n_features = side_length^2;
Y = zeros(n_examples, 1);
X = zeros(n_examples, n_features);
cp1 = 1
for count=1:n_examples
    labelFn = char(labelList(count));
    imageFn = char(imageList(count));
    
    L = load(labelFn);
    % get gender, first element
    Y(count) = L(1);
    
    im = rgb2gray(imread(imageFn));
    im = imresize(im, [side_length, side_length]);
    X(count,:) = im(:); % flatten
    
end
cp2 = 1
csvwrite([split, 'Feats.csv'], X)
csvwrite([split, 'Labels.csv'], Y)