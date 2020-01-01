% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)

zip_train = readmatrix('zip_train.csv');
zip_test = readmatrix('zip_test.csv');

fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip_train(find(zip_train(:,1) == 1 | zip_train(:,1) == 3),:);
X_tr = subsample(:,2:257);
y_tr = subsample(:,1);
subsample = zip_test(find(zip_test(:,1) == 1 | zip_test(:,1) == 3),:);
X_te = subsample(:,2:257);
y_te = subsample(:,1);

ct = fitctree(X_tr, y_tr, 'CrossVal', 'on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
t = fitctree(X_tr, y_tr);
fprintf('The test error of decision trees is %.4f\n', sum(predict(t,X_te) ~= y_te)/length(y_te));
[oobErr_bagTrees, testErr_bagTrees] = BaggedTrees(X_tr, y_tr, X_te, y_te, 500);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', oobErr_bagTrees);
fprintf('The test error of 200 bagged decision trees is %.4f\n', testErr_bagTrees);
[oobErr_randForest, testErr_randForest] = RandomForest(X_tr, y_tr, X_te, y_te, 500, floor(size(X_tr, 2)/3));
fprintf('The OOB error of the random forest is %.4f\n', oobErr_randForest);
fprintf('The test error of the random forest is %.4f\n', testErr_randForest);

fprintf('\n');

fprintf('Now working on the three-vs-five problem...\n\n');
subsample = zip_train(find(zip_train(:,1) == 3 | zip_train(:,1) == 5),:);
X_tr = subsample(:,2:257);
y_tr = subsample(:,1);
subsample = zip_test(find(zip_test(:,1) == 3 | zip_test(:,1) == 5),:);
X_te = subsample(:,2:257);
y_te = subsample(:,1);

ct = fitctree(X_tr, y_tr, 'CrossVal', 'on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
t = fitctree(X_tr, y_tr);
fprintf('The test error of decision trees is %.4f\n', sum(predict(t,X_te) ~= y_te)/length(y_te));
[oobErr_bagTrees, testErr_bagTrees] = BaggedTrees(X_tr, y_tr, X_te, y_te, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', oobErr_bagTrees);
fprintf('The test error of 200 bagged decision trees is %.4f\n', testErr_bagTrees);
[oobErr_randForest, testErr_randForest] = RandomForest(X_tr, y_tr, X_te, y_te, 200, floor(size(X_tr, 2)/3));
fprintf('The OOB error of the random forest is %.4f\n', oobErr_randForest);
fprintf('The test error of the random forest is %.4f\n', testErr_randForest);

% one vs three problem: 3 becomes -1 and 1 remains 1
y_tr(y_tr == 3)=-1;
y_te(y_te == 3)=-1;

% three vs five problem: 3 becomes -1 and 5 becomes 1
y_tr(y_tr == 3)=-1;
y_te(y_te == 3)=-1;
y_tr(y_tr == 5)= 1;
y_te(y_te == 5)= 1;

d=200;
train_ada_algo = zeros(d,1);
test_ada_algo = zeros(d,1);
e_train = zeros(d,1);

for k = 1:d
    [train_ada_algo(k,1), test_ada_algo(k,1)]= AdaBoost(X_tr, y_tr, X_te, y_te, k);
end

indices = 0:10:200;
indices(1)  = 1;

plot(indices, test_ada_algo(indices))
titlecaption = sprintf('AdaBoost: One vs. Three Problem \n Training and Testing Error as a Function of Number of Weak Learners');

title(titlecaption, 'FontSize', 14)
xlabel('Number of Weak Learners', 'FontSize', 14)
ylabel('Error', 'FontSize', 14)
hold on
plot(indices, train_ada_algo(indices))
legend('Testing Error','Training Error', 'FontSize', 14)


