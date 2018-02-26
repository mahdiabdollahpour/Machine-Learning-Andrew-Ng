function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
[t1,t2] = size(all_theta);
[x1,x2] = size(X);

disp('theta rows');
disp(t1);
disp('theta cols');
disp(t2);
disp('X rows');
disp(x1);
disp('X cols');
disp(x2);



perdic = X * transpose(all_theta);

% perdic = zeros(x1,1);
[per1,per2]  = size(perdic);
disp(perdic);
for i=1:x1
    index = 1;
    val = 0;
    for j=1:per2
       if perdic(i,j)> val
          val = perdic(i,j);
          index = j;
       end
    end
   p(i,1) = index;
    
end


% =========================================================================


end
