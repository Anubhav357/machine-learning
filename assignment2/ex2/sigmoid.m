function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================
n=size(z);
m=n(2);
n=n(1);
for i=1:n,
  for j=1:m,
    temp=(1+e^(-z(i,j)));
    g(i,j)=1/temp;
  endfor
end
