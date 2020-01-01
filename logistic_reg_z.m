function [t, w, e_in] = logistic_reg_z(X, y, w_init, max_its, eta)

% logistic_reg: learn a logistic regression model using gradient descent
%  Inputs:
%       X:       data matrix (without an initial column of 1s)
%       y:       data labels (plus or minus 1)
%       w_init:  initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta:     learning rate
%     
%  Outputs:
%        t:    the number of iterations gradient descent ran for
%        w:    learned weight vector
%        e_in: in-sample (cross-entropy) error

% gradient = sigma (-yixi/exp(yiWtxi+1)
% while t<max_its
% wt+1 = wt - eta*gradient
% ein = 1/n sigma (ln(1+exp(-yiwTxi))

t = 0;
w = w_init;
N = size(X,1);
d = size(X,2);
gradient_t = ones(1,d);
tic
while (sum(abs(gradient_t) < .000001) < d)
     if t == 0
        gradient_t = 0;
    end
    
    for j=1:N
        gradient_t = gradient_t + (y(j).*X(j,:))./(1+exp(y(j)*(w'*X(j,:)')'));
    end
    gradient_t = (-1/N).*gradient_t;
    
    %gradient_t = (-1/N)*sum(y.*X/(1+exp(y'*(w'*X')')))
    %gradient_t = (-1/N).*((y'*X')/(1+exp(y'*(w'*X)')))
    v_t = -gradient_t';
    w = w + eta.*v_t;
    t = t + 1;
end
elapsedTime = toc
e_in = 0;

for i=1:N
    e_in = e_in + log(1+exp(-y(i)'*(w'*X(i,:)')'));
end
e_in = (1/N)*e_in;
end
