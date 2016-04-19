function heatmap(N, siz)
% plots a heatmap of the N matrix
% N: shape (100, 4)
% siz: shape of the grid (defaults to 10x10)

if nargin < 2
    siz = [10 10];

% count how many times it has had to make decisions in each state
counts = sum(N, 2);
% counts = 1:100; % JUST A TEST

% plot
colormap('default');
imagesc(reshape(counts, siz));
% caxis([0 14*1e4]);
colorbar;

end