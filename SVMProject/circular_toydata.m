grid_step = 0.17;
lim = 2;
x1 = -lim:grid_step:lim;
x2 = -lim:grid_step:lim;
[X1,X2] = ndgrid(x1, x2);
noise = 0.1 * randn(size(X1));
out = (X1.^2 + X2.^2) + noise < 1;
gscatter(X1(:),X2(:),out(:))

data = [X1(:), X2(:)]';
label = (out(:) - 0.5) .* 2; % -1, 1

