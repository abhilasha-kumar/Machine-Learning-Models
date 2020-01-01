function [train_err, test_err] = AdaBoost(X_tr, y_tr, X_te, y_te, numTrees)
% AdaBoost: Implement AdaBoost with decision stumps as the weak learners. 
%           (hint: read the "Name-Value Pair Arguments" part of the "fitctree" documentation)
% need to use the "Weights" argument here
%   Inputs:
%           X_tr: Training data
%           y_tr: Training labels
%           X_te: Testing data
%           y_te: Testing labels
%           numTrees: The number of trees to use
%  Outputs: 
%           train_err: Classification error of the learned ensemble on the training data
%           test_err: Classification error of the learned ensemble on test data
% 
% You may use "fitctree" but not any inbuilt boosting function
% initialize w
w = (1/length(y_tr))*ones(length(y_tr),1);

for t = 1:numTrees
    e_t(t) = 0;
    % learn a decision tree tb on Db that minimizes the weighted training error 
    t_b = fitctree(X_tr, y_tr,'MinParentSize',size(X_tr,1),'Prune','off','MergeLeaves','off', 'Weights', w);
    %learn h_t for tree t for training data
    h_t(:,t) = predict(t_b,X_tr); 
    
    %learn h_t for tree t for testing data
    h_t_test(:,t) = predict(t_b,X_te); 
    
    %compute weighted training error
    for j = 1:length(y_tr)
        e_t(t) = e_t(t) + w(j)*(h_t(j,t) ~= y_tr(j)); %keep track of this error: this is the training error? 
    end
    %calculate importance
    importance_t(t) = (0.5)*log((1-e_t(t))/e_t(t));
    Z_t = 2*sqrt(e_t(t)/(1-e_t(t)));
    %update weights
    for z = 1:length(w)
        w(z) = (1/Z_t)*w(z) * exp(-importance_t(t)*y_tr(z)*h_t(z,t));
    end
end

%compute aggregate predictions for training data
predictions_train = zeros(size(X_tr,1),1);
for k = 1:size(X_tr,1)
    sum_product_train = 0;
    for t = 1:length(importance_t)
        sum_product_train = sum_product_train + importance_t(t)*h_t(k,t);
    end
    predictions_train(k) = sign(sum_product_train);
end

train_err = sum(predictions_train ~= y_tr)/length(y_tr);
%compute aggregate predictions for test data
predictions_test = zeros(size(X_te,1),1);
for m = 1:size(X_te,1)
    sum_product_test = 0;
    for t = 1:length(importance_t)
        sum_product_test = sum_product_test + importance_t(t)*h_t_test(m,t);
    end
    predictions_test(m) = sign(sum_product_test);
end

test_err = sum(predictions_test ~= y_te)/length(y_te);

end