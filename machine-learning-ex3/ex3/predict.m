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

% add ones to each row
X = [ones(size(X,1),1) X];

% calculate first hidden layer for all m - 
% produce a matrix of theta1*a1 for all m - m by size(a2)
% X - m by n
% Theta1 - size(a2) by n
A2 = sigmoid(X * Theta1');

A2 = [ones(size(A2,1),1) A2];

% calculate the second layer - output classifiers
% produce a matrix of theta2*a2 for all m - m by size(a3)
% size(a3) equals theta2
A3 = sigmoid(A2 * Theta2');

% collapse the classifiers into single digit output
% storing the index containing highest probability in each row of p
[Z, p] = max(A3, [], 2);

% =========================================================================


end
