function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%num_labels
for i=1:m
  in_layer=X(i,:);
  in_layer=[1,in_layer];
  
  mid_neuron=in_layer*Theta1';
  mid_neuron=sigmoid(mid_neuron);
  mid_neuron=[1,mid_neuron];

  final_layer=mid_neuron*Theta2';
  final_layer=final_layer';
  final_layer=sigmoid(final_layer);

  ans_expected=zeros(num_labels,1);
  ans_given=y(i);
  ans_expected(ans_given)=1;
  
  log1=log(final_layer);
  log2=log(1-final_layer);
  ans_expected2=1-ans_expected;
  ans_mat=ans_expected.*log1+ans_expected2.*log2;
  addend=sum(ans_mat);
  addend*=-1;
  J+=addend;
endfor 

J=J/m;
size(nn_params);
nn_params;

square_of_weights=nn_params.*nn_params;
sum_parameters=sum(square_of_weights);


sub_mat=Theta1(:,1);
sq=sub_mat.*sub_mat;
sum_parameters-=sum(sq);


sub_mat=Theta2(:,1);
sq=sub_mat.*sub_mat;
sum_parameters-=sum(sq);

J+=(lambda/(2*m))*sum_parameters;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


for i =1:m
  a1=X(i,:);
  size(a1);
  a1=[1,a1];
  z2=a1*Theta1';
  a2=sigmoid(z2);
  a2=[1,a2];
  z3=a2*Theta2';
  a3=sigmoid(z3);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  a3=a3';
  a2=a2';
  z2=z2';
  z3=z3';
  a1=a1';
  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  ans_exp=zeros(num_labels,1);
  ans_given=y(i);
  ans_exp(ans_given)=1;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sd3=a3-ans_exp;
  %fprintf("size of theta2 is ");
 % size(Theta2)
  %fprintf("size of sd3 is ");
 % size(sd3)
 % fprintf("size of z2 is ");
 % size(z2)
  sd2=Theta2'*sd3.*[1;sigmoidGradient(z2)];
  sd2=sd2(2:end);
%  fprintf("size of sd2 is ");
 % size(sd2)
 % fprintf("size of a2 is ");
 % size(a2)
  Theta2_grad=Theta2_grad+sd3*a2';
  Theta1_grad=Theta1_grad+sd2*a1';
  %%%%%%%%%%%%%%%%%%%%%%%
endfor
Theta1_grad=(1/m)*Theta1_grad;
Theta2_grad=(1/m)*Theta2_grad;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
add=Theta1*(lambda/m);
add(:,1)=0;
Theta1_grad+=add;

add=Theta2*(lambda/m);
add(:,1)=0;
Theta2_grad+=add;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
