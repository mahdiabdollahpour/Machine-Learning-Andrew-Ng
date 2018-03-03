function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
[n1 n2]= size(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

for i=1:m 
   J = J + ( X(i,:) * theta - y(i))^2;
end
reg = 0;
for i=2:n1
   reg = reg + ( theta(i) )^2; 
end
reg = reg * lambda;
J = (J + reg)*(1/(2*m));
grad(1) = 0;
for i=1:m 
   grad(1) = grad(1) + ( X(i,:) * theta - y(i) ) * X(i,1);
end
grad(1) = grad(1)/m;

for j=2:n1 
   grad(j) = 0;
   for i=1:m 
       grad(j) = grad(j) + ( X(i,:) * theta - y(i) ) * X(i,j);
   end 
   grad(j) = grad(j) + (lambda) * theta(j);
   grad(j) = grad(j)/m;
end










% =========================================================================

grad = grad(:);

end
