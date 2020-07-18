function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
#ones_r=ones(m,1);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
tmp=theta'*X';
tmp=sigmoid(tmp);
tmp=tmp';
#X
#tmp
f1=@(x) log(x);
f2=@(x) log(1-x);
p1=f1(tmp);
p2=f2(tmp);
k1=y.*(p1);
k2=(1-y).*p2;
s1=sum(k1);
s2=sum(k2);
J=(-1/m)*(s1+s2);
tmp=tmp-y;
tmp=X'*tmp;
%grad=sum(tmp);
grad=(1/m).*tmp;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
