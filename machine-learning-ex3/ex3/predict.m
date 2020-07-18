function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
#size(X)
#size(Theta1)
#size(Theta2)

a1= [ones(rows(X), 1) X];
Theta1=Theta1';
mid=a1*Theta1;
mid=sigmoid(mid);
mid=[ones(rows(mid),1) mid];
Theta2=Theta2';
check=mid*Theta2;
check=sigmoid(check);
#size(check)

for i=1:m
  max_val=-1;
  t=1;
  for j=1:num_labels
    if check(i,j)>max_val
      max_val=check(i,j);
      t=j;
    endif
  endfor
  p(i,1)=t;
endfor

     





% =========================================================================


end
