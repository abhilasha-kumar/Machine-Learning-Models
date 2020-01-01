M = readtable('cleveland_train.csv','ReadRowNames',false)
M = M{:,:}
%change from 0/1 to -1/1
idx = M(:,14) == 0;
M(idx,14) = -1;

X = M(:,1:13);
y = M(:,14);
N = size(X, 1)
w_init = zeros(14,1);
leading1 = ones(N,1);
data_in = horzcat(leading1, X);
max_its = 10000
eta = .00001
x_demo = [1 3 2 1; 1 1 2 3]
y_demo = [1 -1]'
w_demo = zeros(4,1)
[t w e_in] = logistic_reg(data_in, y, w_init, max_its, eta)



% test data

testdata = readtable('cleveland_test.csv','ReadRowNames',false);
testdata = testdata{:,:};
%change from 0/1 to -1/1
idx = testdata(:,14) == 0;
testdata(idx,14) = -1;
N_test = size(X_test, 1);
leading1_test = ones(N_test,1);
X_test = testdata(:,1:13);
y_test = testdata(:,14);
data_test = horzcat(leading1_test, X_test);

binary_error_test = find_test_error(w, X_test, y_test)
binray_error_train = find_test_error(w, X, y)

% scaling the features
x_zscored = zscore(X)

data_in_z = horzcat(leading1, x_zscored);
max_its = 10000
eta = .001
[t_z w_z e_in_z] = logistic_reg_z(data_in_z, y, w_init, max_its, eta)

% normalizing the test data too 
xtest_zscored = zscore(X_test)
binary_error_test = find_test_error(w_z, X_test, y_test)

