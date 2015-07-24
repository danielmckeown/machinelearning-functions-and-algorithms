function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.


g = zeros(size(z));




g = 1 ./ ( 1 + exp(-z));

%  ./ makes it so that the sigmoid function is performed on every element

% =============================================================

end
