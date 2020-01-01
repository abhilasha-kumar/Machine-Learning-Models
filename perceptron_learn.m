function [w, iterations] = perceptron_learn(data_in)
% perceptron_learn: Run PLA on the input data
% and count how long it takes to get to the correct labels
% PLA works by first looking at any one misclassified eg.
% and updating its w = w+yx
% Inputs:  data_in is a matrix with each row representing an (x,y) pair;
%                 the x vector is augmented with a leading 1,
%                 the label, y, is in the last column
% Outputs: w is the learned weight vector; 
%            it should linearly separate the data if it is linearly separable
%          iterations is the number of iterations the algorithm ran for

%initialize iterations to 0 and create zero weight vector
iterations = 0;
w = zeros(11,1);
% get xi and y out from the input data
xi = data_in';
yi = data_in(:,12);
xi(12,:) = [];
%compute the sum wTxi again
sum_w_xi = w'*xi;
%calculate hx i.e., hypothesis signs/outputs
hx = sign(sum_w_xi)';
% while hx is not equal to yi, i.e. there is a misclassified eg
while(isequal(hx, yi) == 0)
    % increase iterations
    iterations = iterations + 1;
    % compute indices for all examples that are misclassified
    misclassified_indices = find(hx ~= yi);
    % pick a random index that is misclassified
    r = datasample(misclassified_indices,1);
    if (hx(r) ~= yi(r)) %this should always be true
        % add yxi to w
        w = w + yi(r)*xi(:,r);
        % compute sum again
        sum_w_xi = w'*xi;
        % compute signs again so it can be compared to yi again
        hx = sign(sum_w_xi)';
    end    
end

end