
% It's time to introduce regualarization into the cost function for logistic regression

function [J, grad] = costFunctionReg(theta, X, y, lambda)

% We will compute the cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Set up for training examples
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


% this will compute the cost of a particular choice of theta(a particular parameter value)
%J = cost.
% Compute the partial derivatives and set grad to the partial
% derivatives of the cost w.r.t. each parameter in theta




[J , grad ] = costFunction(theta, X , y );

penalty = sum(theta( 2: end ) .^ 2 );


J = J + lambda / (2* m ) * penalty ;


grad (2: end ) = grad ( 2 : end ) +  (lambda / m ) * theta (2:end);





% =============================================================

end
