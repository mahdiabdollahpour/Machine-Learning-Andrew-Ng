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
reg = 0;

z = X * theta;
for i=1:m
    J = J + y(i) * log(sigmoid(z(i)+(1-y(i)))) + (1 - y(i)) * log(1-sigmoid(z(i)));
end
J = (-1) * J / m;

for i=2:size(theta)
      reg = reg + theta(i) * theta(i); 
end
reg = reg * lambda / ( 2 * m );

J = J + reg;
 for j=1:m 
        grad(1) = grad(1) + ( sigmoid(z(j)) - y(j) )* X(j,1) ;
    end
grad(1) = grad(1) / m;
    
for i=2:length(grad)
    for j=1:m 
        grad(i) = grad(i) + ( sigmoid(z(j)) - y(j) )* X(j,i) ;
    end
    grad(i) = grad(i) / m; 
    grad(i) = grad(i) + + (lambda / m) * theta(i);
end







% =============================================================

end
