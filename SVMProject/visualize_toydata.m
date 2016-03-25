function visualize_toydata(trX, trY, svm)
    hold on
    gscatter(trX(1, :), trX(2, :), trY)
    
    grid_step = 0.03;
    x1 = -2:grid_step:2;
    x2 = -2:grid_step:2;
    [X1,X2] = ndgrid(x1,x2);
    Xbound = [X1(:), X2(:)]';
    g_randompts = svm.discriminate(Xbound);
    bound_idx = find(abs(g_randompts) < 0.05);
    
    % plot support vectors
    plot(svm.sv_data(1,:), svm.sv_data(2,:), 'm*');
    
    % plot decision boundary
    plot(Xbound(1, bound_idx), Xbound(2,bound_idx), 'b.');
    hold off
end