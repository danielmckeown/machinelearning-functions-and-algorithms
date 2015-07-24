
% First I define a function that takes as arguement J and grad

function [J, grad] = costFunction(theta, X, y)
% This COSTFUNCTION is used to Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(theta , 1 );

% this code will Compute the cost of a particular choice of theta.
% J is equal to the cost.
% we will compute the partial derivatives and set grad to the partial
% derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


h = sigmoid(X*theta);


J = ( ( -y )'* log(h) - (1 - y )' * log(1 - h ) ) / m;



for i = 1 : m,
    highfivex = sigmoid(theta'* X(i,:)');
    timely = highfivex - y(i);
    for j=1:n,
       grad(j) = grad(j) + timely*X(i,j);
    end;
    
 end;
 
 grad = grad / m;




% =============================================================

end
