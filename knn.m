N1 = 3; N2 = 3; % Class sizes
%x = [2 5 6 0 2 4;0 2 3 1 3 4]';
x = [2 5 6 0 2 4;0 10 15 5 15 20]';

t = [repmat(0,N1,1);repmat(1,N2,1)];
N = size(x,1);
ma = {'ko','ks'};
fc = {[0 0 0],[1 1 1]};
tv = unique(t);
figure(1); hold off
for i = 1:length(tv)
    pos = find(t==tv(i));
    plot(x(pos,1),x(pos,2),ma{i},'markerfacecolor',fc{i}, 'MarkerSize',15);
    hold on
end

[Xv Yv] = meshgrid(min(x(:,1)):0.1:max(x(:,1)),min(x(:,2)):0.1:max(x(:,2)));
% Loop over test points

Kvals = [1];
for kv = 1:length(Kvals)
    classes = zeros(size(Xv));
    K = Kvals(kv);
    for i = 1:length(Xv(:))
        this = [Xv(i) Yv(i)];
        dists = sum((x - repmat(this,N,1)).^2,2);
        [d I] = sort(dists,'ascend');
        [a,b] = hist(t(I(1:K)));
        pos = find(a==max(a));
        if length(pos)>1
            order = randperm(length(pos));
            pos = pos(order(1));
        end
        classes(i) = b(pos);
    end
    figure(1); hold off
    for i = 1:length(tv)
        pos = find(t==tv(i));
        plot(x(pos,1),x(pos,2),ma{i},'markerfacecolor',fc{i},'MarkerSize',15);
        hold on
    end
    contour(Xv,Yv,classes,[0.5 0.5],'k')
    ti = sprintf('Decision Boundary for K = %g',K);
    title(ti, 'FontSize', 14);
    xlabel('x1', 'FontSize', 14)
    ylabel('x2', 'FontSize', 14)
end