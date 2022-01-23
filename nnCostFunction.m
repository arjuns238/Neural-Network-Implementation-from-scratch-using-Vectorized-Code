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
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. Theta1_grad and Theta2_grad return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2, respectively. 
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K.  
%
% Part 3: Implement regularization with the cost function and gradients.
%

X = [ones(m,1) X];
a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1), a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
hyp = a3;
y_Vec = (1:num_labels)==y;

% Cost function
J = (1/m)*sum((sum(((-y_Vec).*log(hyp))-((1-y_Vec).*log(1-hyp)))));
J = J + (lambda/(2*m))*...
 (sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
 
%Backprop
for t = 1:m
%Layer 1
  a1 = X(t,:)';
%Layer 2 
  z2 = Theta1 * a1;
  a2 = [1;sigmoid(z2)];
%Layer 3 hypothesis
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  yVector = y_Vec(t,:)';
  delta3 = a3 - yVector;
  delta2 = (Theta2' * delta3).*[1;sigmoidGradient(z2)];
  delta2 = delta2(2:end);
  Theta1_grad = Theta1_grad + (delta2 * a1');
  Theta2_grad = Theta2_grad + (delta3 * a2');
  
endfor
Theta1_grad(:,1) = ((1/m)*(Theta1_grad(:,1)));
Theta1_grad(:,2:end) = ((1/m)*(Theta1_grad(:,2:end))) + ((lambda/m)...
 * Theta1(:,2:end));
Theta2_grad(:,1) = (1/m)*(Theta2_grad(:,1));
Theta2_grad(:,2:end) = ((1/m)*(Theta2_grad(:,2:end))) + ((lambda/m)...
 * Theta2(:,2:end));


  
  
   





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
