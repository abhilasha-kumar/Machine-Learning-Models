function [oob_err, test_err] = BaggedTrees(X_tr, y_tr, X_te, y_te, numBags)
% BaggedTrees: Learns an ensemble of numBags CART decision trees on the input dataset 
%              and also plots the out-of-bag error as a function of the number of bags
%      Inputs:
%              X_tr: Training data
%              y_tr: Training labels
%              X_te: Testing data
%              y_te: Testing labels
%              numBags: Number of trees to learn in the ensemble
%     Outputs: 
%	           oob_err: Out-of-bag classification error of the final learned ensemble
%              test_err: Classification error of the final learned ensemble on test data
%
% You may use "fitctree" but not "TreeBagger" or any other inbuilt bagging function
D = horzcat(y_tr, X_tr);
% create a dataset to keep track of predictions of each tree
oob_predictions = zeros(size(D,1),numBags);
for i=1:numBags
    % get a random sample of n = size(D,1) points from 1:size(D,1)
    % with replacement
    % these will be the indices we can use later
    indices = datasample(1:size(D,1),size(D,1));
    % create D_b by using these indices
    D_b = D(indices,:);
    % divide D_b into x and y
    x_tr_b = D_b(:,2:size(D,2));
    y_tr_b = D_b(:,1);
    % learn a decision tree tb on Db
    t_b = fitctree(x_tr_b, y_tr_b);
    % oob set is the set with indices that are in D but not in D_b
    % as this is the "out of bag" set for current tree
    
    oob_indices = setdiff(1:size(D,1), unique(indices));
    
    % next we want to add in predictions from t_b into oob_predictions
    % only for the oob_indices points, everything else remains 0
    
    oob_predictions(oob_indices,i) = predict(t_b, D(oob_indices,2:size(D,2)));
          
    % store predictions for test data for each tree also
    % each tree is a row here (i)
    test_predictions(i,:) = predict(t_b, X_te);
end

oob_predictions_final = oob_predictions;
%learning an ensemble decision tree

% now we have oob predictions for each training point from each tree
% in oob_predictions
% to compute oob_error as a fun of numBags
% we calculate oob error for 1 bag, then 1+2nd bag and so on using
% plurarity vote

for t = 1:size(oob_predictions_final,2)
    for p = 1:size(oob_predictions_final,1)
        point_predictions = oob_predictions(p,1:t);
        % find set t_p that did not use p for training
        % this will be the columns that are non zero for p 
        t_p_indices = find(point_predictions ~= 0);
        % find plurality prediction for this point at current iteration
        % if t_p_indices is not empty
        if(~isempty(t_p_indices))
            t_p_prediction(p) = mode(oob_predictions(p,t_p_indices));
        else
            t_p_prediction(p) = 100;
        end
    end
    
    
    % now we get the oob error from those points that are ~= 100
    nonzeroindices = find(t_p_prediction ~= 100);
    
    nonzeropoints = t_p_prediction(nonzeroindices);
    oob_error(t) = sum(nonzeropoints' ~= y_tr(nonzeroindices))/length(y_tr(nonzeroindices));
end
    

 % TEST ERROR
 test_predictions_final = test_predictions';
  % npw each row of test_predictions is a pt and each col is a tree
   % plurality vote for test predictions 
  test_error_mode_predictions = mode(test_predictions_final,2);
  classification_error = (test_error_mode_predictions ~= y_te);
  test_err = sum(classification_error)/length(y_te);
  
% for t = 1:size(test_predictions_final,2)
%     for p = 1:size(test_predictions_final,1)
%         %get 1 to t trees
%         testpoint_predictions = test_predictions_final(p, 1:t);
%         %take plurality vote for those trees
%         test_plurality(p) = mode(test_predictions_final(p, testpoint_predictions));
%     end
%      %now for t trees, we have the plurality vote for all test points
%      % now calc test error for t trees
%     test_error(t) = sum(test_plurality' ~= y_te)/length(y_te);
% end

% first we can plot the oob error and test error as a function of trees
plot(1:length(oob_error),oob_error)
title('Bagging: Out of Bag Error as a Function of Number of Bags', 'FontSize', 14)
xlabel('Number of Bags', 'FontSize', 14)
ylabel('OOB error', 'FontSize', 14)
% hold on
% plot(1:length(test_error),test_error)
% legend('OOB Error','Testing Error', 'FontSize', 14)

 % next, we need to return the final value of oob error from the final
 % learned ensemble, i.e., when ALL bags are considered
 % this is the last value in oob_error_bags
 
oob_err = oob_error(length(oob_error));
 
%we simply return the final test error at the end
%test_err = test_error(length(test_error));

end