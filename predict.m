function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples


p = zeros(m, 1);
%    set p to a vector of 0's and 1's


%use a for loop to loop through the training data set



for i = 1: m,
    h = sigmoid(X(i,:) * theta);
    if h >= 0.5,
       p(i) = 1;
    else
       p(i) = 0;
       
    end

end

% here if our hypothesis is greater than or equal to 0.5
%then we predict that the training data point is 1 else if 
%it is less than 0.5 then we predict that the hypothesis is 0
% =========================================================================


end
