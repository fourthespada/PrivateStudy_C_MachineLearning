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

% > feedforward to calculate activations

%   calculate hidden layer
%   produce a matrix of theta1*a1 for all m - m by size(a2)
%   X - m by n
%   Theta1 - size(a2) by n
X = [ones(m, 1) X];
Z2 = X * Theta1';
A2 = sigmoid(Z2);

%   calculate output layer
%   produce a matrix of theta2*a2 for all m - m by size(a3)
%   size(A3) equals theta2
A2 = [ones(size(A2,1),1) A2];
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

%   calculate cost by looking at each classifier
%   do not collapse classifiers into single digit output
%   decode y from m by 1 to m by num_labels
%Y = zeros(m, num_labels);
%for i=1:m
%    Y(i, y(i)) = 1;
%end
% better:
Y = eye(num_labels);
Y = Y(y,:); 
 
% parameter wise multiplication of Y and hyp produces a matrix size m by K
% have to sum twice to sum over all K classes and m examples 
% unregularized
J = 1/m * sum(sum(-Y .* log(A3) - ((1-Y) .* log(1-A3))));

% add regularization to cost
Theta1_unbiased = Theta1(:,2:end);
Theta2_unbiased = Theta2(:,2:end);

J = J + (lambda / (2*m)) * ... 
    (sum(sum((Theta1_unbiased .^ 2))) + sum(sum((Theta2_unbiased .^ 2))));

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

% compute d3 for all examples: d3 is m by K (examples by output units)
d3 = A3 - Y;

% compute d2 for all examples: d2 is m by h1 (examples by hidden layer units)
T2_noBias = Theta2(:,2:end);
d2 = (d3 * T2_noBias) .* sigmoidGradient(Z2);

D2 = d3' * A2;
D1 = d2' * X;

% these arrays are (n by h1+1) and (h1 by K+1)
Theta1_grad = D1 ./ m;
Theta2_grad = D2 ./ m;

% regularization
T1_regterm = [zeros(hidden_layer_size, 1) Theta1(:,2:end)];
T2_regterm = [zeros(num_labels, 1) Theta2(:,2:end)];
T1_regterm = T1_regterm .* (lambda/m);
T2_regterm = T2_regterm .* (lambda/m);
Theta1_grad = Theta1_grad + T1_regterm;
Theta2_grad = Theta2_grad + T2_regterm;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
