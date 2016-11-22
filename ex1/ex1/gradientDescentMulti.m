function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
cs = size(X, 2);    


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
#    for c = 1: cs;
#     
#        a = theta(c,1)
#        b = (alpha/m) 
#        d = X(:,c);      
#        cc = sum((X * theta - y).*d)
#        e = b * cc;
#        f = e .* d;
#        g = a -f;
     
#     thetaNew(c) = theta(c) - (alpha / m) * sum(((X * theta ) - y) .* X(:,c));        
 

  
 
   # end
  #theta = thetaNew
  #  theta = theta - (alpha/m) * (X') * (X * theta - y);
      delta = ((theta' * X' - y')*X)';
    theta = theta - alpha / m * delta;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
