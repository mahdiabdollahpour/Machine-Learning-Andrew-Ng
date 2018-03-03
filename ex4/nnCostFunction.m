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
[xrow, xcol] = size(X);         
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

myY = zeros(m,num_labels);
for i=1:m
    myY(i,y(i))=1;
end
aa1 = [ones(xrow,1)  X];
% aa1 = X;
zz2 = aa1 * transpose(Theta1);
aa2 = sigmoid(zz2);
[a2row , a2col] = size(aa2);
aa2 = [ones(a2row,1) aa2];
zz3 = aa2 * transpose(Theta2);
aa3 = sigmoid(zz3);


for i=1:m
   for j=1:num_labels
      J = J + ( (-1)* myY(i,j) * log( aa3(i,j)) - (1 - myY(i,j) ) * log( 1 - aa3(i,j) )); 
   end    
end

J = J/m;

[t1row t1col] = size(Theta1);
reg = 0;
for i=1:t1row
   for j=2:t1col
        reg = reg + Theta1(i,j) * Theta1(i,j);    
   end
end

[t2row t2col] = size(Theta2);
for i=1:t2row
   for j=2:t2col
        reg = reg + Theta2(i,j) * Theta2(i,j);    
   end
end
reg = reg *(lambda/(2*m));
J = J + reg;

a1 = [ones(m, 1) X];
for t = 1:m
	a1 = [1; X(t,:)'];
    disp(size(a1));
    disp(size(Theta1));
	z2 = Theta1 * a1;
% 	disp(size(z2));
     a2 = [1;sigmoid(z2)];
	disp(size(Theta2'));
   disp(size(a2));
    z3 = Theta2 * a2;
	a3 = sigmoid(z3);
    disp('///////');
    disp(size(a3));
    disp(size(myY(t,:)'));
    delta3 = a3 - myY(t,:)';
	delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
	delta2 = delta2(2:end);
	Theta1_grad = Theta1_grad + delta2 * a1';
	Theta2_grad = Theta2_grad + delta3 * a2';
end
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
