%generate input space x1x2 in [-1,1]
rng('default');
rng(2);

range1 = -1;
range2 = 1;

for k = 1:1000000
    
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

    % finding range of values of a for this data that separate it
    % so values of a s.t. sign(x2 -a )= sign(x2)
    % then for a_random, we will pick any a from that set

    a = (range2-range1).*rand(10000,1) + range1;
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
    a_random(k) = datasample(separating_a,1);
    
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

    D_midpoint_x2 = (D_positive_closest(1,2) + D_negative_closest(1,2))/2;

    % now for these 2 points find the a that has max distance

    for  i = 1:length(separating_a)
        % for each a calculate distance to each training point
        separating_a(i,2) = abs(D_midpoint_x2 - separating_a(i));
    end

    a_svm(k) = separating_a(separating_a(:,2) == min(separating_a(:,2)));

end

grid on;
xlim([-1,1]);
hold on
nbins = 25
histogram(a_random, nbins)
title('Histogram of a-random from 1000000 runs', 'FontSize', 14)
xlabel('Value of a', 'FontSize', 14)
ylabel('Frequency', 'FontSize', 14)
hold on

xlim([-1,1]);
hold on
histogram(a_svm, nbins)
title('Histogram of a-SVM from 1000000 runs', 'FontSize', 14)
xlabel('Value of a', 'FontSize', 14)
ylabel('Frequency', 'FontSize', 14)
legend('Random a','SVMa', 'FontSize', 14)

N = 3
L = 2^N;
T = zeros(L,N);
for i=1:N
   temp = [-ones(L/2^i,1); ones(L/2^i,1)];
   T(:,i) = repmat(temp,2^(i-1),1);
end

x1 = T(:,1);
x2 = T(:,2);
x3 = T(:,3);

N = 3
L = 2^N;
T2 = zeros(L,N);
for i=1:N
   temp = [zeros(L/2^i,1); ones(L/2^i,1)];
   T2(:,i) = repmat(temp,2^(i-1),1);
end

x1_2 = T2(:,1);
x2_2 = T2(:,2);
x3_2 = T2(:,3);

AND_x1_x2_x3 = sign(x1 + x2 + x3 - 1.5)
OR_x1_x2_x3 = sign(x1 + x2 + x3 + 1.5)

AND_x1_x2 = sign(x1 + x2 - 1.5);

P = sign(AND_x1_x2 + x3 + 1.5);
Q = sign(-sign(x1+ x2  - 1.5) - x3  + 1.5);
R = -sign(sign(x1+ x2 -1.5) + x3 - 1.5);

P = (x1 & x2 & ~x3)
Q = ~(x1_2 & x2_2) & x3_2
R = ~x1_2 & ~x2_2 & x3_2

ANSWER = xor(x1_2 & x2_2, x3_2)

POSSIBLE = (x1_2 & x2_2 & ~x3_2) | (~x1_2 & x3_2) | (~x2_2 & x3_2)

POSSIBLE_algerbraic =  sign(sign(x1 + x2 -x3 - 1.5) + sign(-x1 + x3 - 1.5) + sign(-x2 + x3 - 1.5) + 1.5)



