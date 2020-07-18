function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%fprintf("investigating starts")
#X
#y
#lambda
#theta

grad = zeros(size(theta));

tmp=theta'*X';
tmp=sigmoid(tmp);
tmp=tmp';
#X
h=tmp;
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
#J
####################
a1=theta(1:1);
J=J+(lambda/(2*m))*((theta'*theta)-a1*a1);
J
########################
tmp=tmp-y;
tmp=X'*tmp;
%grad=sum(tmp);
grad=(1/m).*tmp
grad+=(lambda/m).*theta;
grad(1:1)=grad(1:1)-(lambda/m)*a1;







% =============================================================

end
