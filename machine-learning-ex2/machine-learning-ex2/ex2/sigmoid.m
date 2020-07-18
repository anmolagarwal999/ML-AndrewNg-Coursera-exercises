function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%f=@(x) x+1;
%f=@(x) x+5;
%g=f(z);
%g=arrayfun(@(x) x/2, z)
func = @(x) (1 ./ (1 .+ e.^((-1).*x))); 
#func = @(x) (x.^2); 
g = func(z);



% =============================================================

end
