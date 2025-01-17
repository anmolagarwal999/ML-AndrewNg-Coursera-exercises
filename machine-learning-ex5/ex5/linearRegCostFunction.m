function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

ans_pred=X*theta;
diff=ans_pred-y;
diff2=diff.*diff;
J=sum(diff2);
theta_sq=theta.*theta;

J+=lambda*(sum(theta_sq)-theta(1)*theta(1));


J/=(2*m);


summation=diff'*X;
summation=summation';
grad=summation;
grad(2:end)+=(lambda)*theta(2:end);
grad/=m;











% =========================================================================

grad = grad(:);

end
