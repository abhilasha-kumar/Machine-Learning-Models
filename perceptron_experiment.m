function [num_iters, bound_minus_ni] = perceptron_experiment(N, d, num_samples)
% perceptron_experiment: Code for running the perceptron experiment in HW1
% Inputs:  N is the number of training examples
%          d is the dimensionality of each example (before adding the 1)
%          num_samples is the number of times to repeat the experiment
% Outputs: num_iters is the # of iterations PLA takes for each sample
%          bound_minus_ni is the difference between the theoretical bound
%                         and the actual number of iterations
%           (both the outputs should be num_samples long)
% bound is R^2||w||2/rho^2
% R = max||xn||
% rho = min(yn.w'xn)
% 
for i=1:num_samples
    %wstar is a column vector
    wstar = unifrnd(0,1, 1,d)';
    %adding a zero for w(0) = 0
    zerovector = zeros(1,1);
    wstar = vertcat(zerovector,wstar);
    %declaring training set from -1 to 1 distribution: this is X
    % each x is also a column vector
    training_set = unifrnd(-1,1, N,d)';
    %adding a leading one to every column vector in X
    leading1 = ones(N,1)';
    data_in = vertcat(leading1, training_set);
    % here we get wTx
    sum_wstar_training_set = wstar'*data_in;
    % yi is the vector that contains the signs from this wTx
    yi = sign(sum_wstar_training_set)';
    % the data going into PLA is training set + yi labels
    % and this data is transposed to make sure the computations work 
    data_in_final = vertcat(data_in,yi');
    [finalweights, iterations] = perceptron_learn(data_in_final');
    num_iters(i,1) = iterations;
    %calculating the theoretical bound
    %element-wise multiply yi with wstar'*data_in
    yi_wstar_xi = yi'.*(wstar'*data_in);
    rho = min(yi_wstar_xi);
    % x is columnwise so we first take norm of each and R = max (normx)
    normxn = vecnorm(data_in);
    R = max(normxn);
    wnorm = norm(wstar);
    bound = ((R^2)*(wnorm^2))/(rho^2);
    bound_minus_ni(i,1) = bound - iterations;
end
end




