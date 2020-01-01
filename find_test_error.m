function [test_error] = find_test_error(w, X, y)

% find_test_error: compute the test error of a linear classifier w. The
%  hypothesis is assumed to be of the form sign([1 x(n,:)] * w)
%  Inputs:
%		w: weight vector
%       X: data matrix (without an initial column of 1s)
%       y: data labels (plus or minus 1)
%     
%  Outputs:
%        test_error: binary error of w on the data set (X, y) error; 
%        this should be between 0 and 1. 

% for every value, first compute sign([1 X]*w) 
N_test = size(X, 1);
leading1_test = ones(N_test,1);
X_final = horzcat(leading1_test, X);

%predictions = sign(1/1+exp(- (w'*X_final'))'-0.5)

predictions = sign(X_final*w);

binary_error = ~(predictions == y);
test_error = sum(binary_error)/size(y,1);
end

