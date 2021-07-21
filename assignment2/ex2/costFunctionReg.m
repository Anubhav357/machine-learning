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






% =============================================================
value=theta'*X';
pval=sigmoid(value);
v1=log(pval);
v2=log(1-pval);
aans=y'.*v1+(1-y').*v2;
sans=sum(aans');
J=sans/m;
J=J*-1;
v3=theta.*theta;
v3=sum(v3)-theta(1)*theta(1);
v3=v3*lambda;
v3=v3/2;
v3=v3/m;
J=J+v3;
grad(1)=(pval-y')*X(:,1)/m;
ct=size(theta);
for i=2:ct,
  grad(i)=((pval-y')*X(:,i))/m+(lambda*theta(i))/m;
endfor
end
