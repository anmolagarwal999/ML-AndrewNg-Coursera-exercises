function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
possibles=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
min_C=-1;
min_sigma=-1;
min_possible=Inf;
sz=8;
for i=1:sz
  for j=1:sz
    C_trial=possibles(i);
    sigma_trial=possibles(j);
    model = svmTrain(X, y, C_trial, @(a, b) gaussianKernel(a, b, sigma_trial));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    
    if error<min_possible
      min_sigma=sigma_trial;
      min_C=C_trial;
      min_possible=error;
    endif
  endfor
endfor

C=min_C;
sigma=min_sigma;






% =========================================================================

end
