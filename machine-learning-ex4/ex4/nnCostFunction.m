function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1:

a1 = [ones(m, 1) X];
a2 = [ones(m, 1) sigmoid(a1 * Theta1')];
h = sigmoid(a2*Theta2'); 
% h = 5000 x 10
% y = 5000 x 1
% y' = 1 x 5000
% yonezero = 5000 x 10
% X = 5000 x 400
% Theta1 = 25 x 401
% Theta2 = 10 x 26

yonezero = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
J = sum(sum((-1 * (yonezero .* log(h)) -(1 - yonezero) .* log(1 - h)) ./ m)); 
J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

%Part 2:

% a1 = 5000 x 401
% a2 = 5000 x 26
% a1t = 401 x 1
% a2t = 26 x 1
% ht = 10 x 1
% yt = 10 x 1
% delta3t = 10 x 1
% z2t = 26 x 1
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
for t=1:m,
    a1t = a1(t,:)';
    a2t = a2(t,:)';
    ht = h(t,:)';
    yt = yonezero(t,:)';
    delta3t = ht - yt;
    z2t = [1; Theta1 * a1t];
    delta2t = (Theta2)'*delta3t.*sigmoidGradient(z2t);
    % The last line : 26 x 1 = 26 x 10 | 10 x 1 .* 26 x 1
    
    % delta2t = 26 x 1
    % delta1 = 25 x 401
    % delta2 = 10 x 26
    
    delta2 = delta2 + delta3t*a2t';
    % 10 x 26 = 10 x 26 + 10 x 1 * 1 x 26
    
    delta1 = delta1 + delta2t(2:end,:)*a1t';
    % 25 x 401 = 25 x 401 + 25 x 1 * 1 x 401
end;

Theta1_grad = (1/m)*delta1 + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m)*delta2 + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:,2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
