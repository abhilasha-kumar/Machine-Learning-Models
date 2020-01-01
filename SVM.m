%generate input space x1x2 in [-1,1]
rng('default');
rng(2);

range1 = -1;
range2 = 1;

% part a
% generate 3 pts at random from upper half of input space
range_x2_1_upper = 0;
range_x2_2_upper = 1;
x1_a_upper = (range2-range1).*rand(3,1) + range1;
x2_a_upper = (range_x2_2_upper-range_x2_1_upper).*rand(3,1) + range_x2_1_upper;

% generate 3 pts at random from lower half of input space
range_x2_1_lower = 0;
range_x2_2_lower = -1;
x1_a_lower = (range2-range1).*rand(3,1) + range1;
x2_a_lower = (range_x2_2_lower-range_x2_1_lower).*rand(3,1) + range_x2_1_lower;

D = [x1_a_upper,x2_a_upper; x1_a_lower,  x2_a_lower];
% now to find a set of values a that separates this data 
D(:,3) = sign(D(:,2));

%plot(D(:,1), D(:,2), 'o','MarkerSize', 12);
% plotting the points
gscatter(D(:,1),D(:,2),D(:,3),'br','xo', 14)
% finding range of values of a for this data that separate it
% so values of a s.t. sign(x2 -a )= sign(x2)
% then for a_random, we will pick any a from that set

a = (range2-range1).*rand(1000,1) + range1;
a(:,2)= 1;

for i=1:length(a)
    for j = 1:size(D,1)
        if(sign(D(j,2)-a(i,1)) ~= sign(D(j,2)))
            % then this value of a is not good anymore
            a(i,2) =0;
            break
        end
    end
end

separating_a = a(a(:,2) == 1);
a_random = datasample(separating_a,1);
g_random = sign(D(:,2)-a_random);
D(:,4) = D(:,2) - a_random



% SVM
% among all the a's need to find the value of a that
% maximizes the distance from the closest training data points
% so for each a run a loop and find the maximum distances
% closest training pts are the ones that are closest to 
% the x2 = 0 line

% first we find those points
% add these distances to D

for i = 1:length(D)
    D(i,4) = sqrt((D(i,1)-D(i,1))^2 + (D(i,2)-0)^2);
end

% now we need to find the "closest" points for each classification (+1/-1)?
D_positive = D(D(:,3) == 1,:);
D_negative = D(D(:,3) == -1,:);

D_positive_closest = D(D(:,4) == min(D_positive(:,4)),:);
D_negative_closest = D(D(:,4) == min(D_negative(:,4)),:);

% the maximal margin separator will be the pt closest to the midpoint of
% these two points

D_midpoint_x2 = (D_positive_closest(1,2) + D_negative_closest(1,2))/2

% now for these 2 points find the a that has max distance

for  i = 1:length(separating_a)
    % for each a calculate distance to each training point
    separating_a(i,2) = abs(D_midpoint_x2 - separating_a(i));
end

a_svm = separating_a(separating_a(:,2) == min(separating_a(:,2)));



random_h_data = [D(:,1), repelem(a_random,6)', repelem(a_svm, 6)']

gscatter(D(:,1),D(:,2),D(:,3),'br','xo', 12);
title('Random and SVM Separator for Dataset D', 'FontSize', 14)
xlabel('x1', 'FontSize', 14)
ylabel('x2', 'FontSize', 14)
hold on
plot(random_h_data(:,1),random_h_data(:,2))
hold on
plot(random_h_data(:,1),random_h_data(:,3))
legend('-1','+1', 'Random a','SVMa', 'FontSize', 14)

