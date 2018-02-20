function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = length(theta);
 temp = zeros(n,1);
    
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

hyp = X * theta; 
    for l=1:n
       
        for i=1:m
            temp(l,1) = temp(l,1) + ((hyp(i) - y(i)) * X(i,l));
        end
        
        temp(l,1) = temp(l,1)/m;
    
   
        temp(l,1) = theta(l,1) - alpha * temp(l,1);
    



    end


    theta = temp;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
