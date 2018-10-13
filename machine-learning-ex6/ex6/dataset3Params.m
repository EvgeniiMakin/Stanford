function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_and_sigma = [0.01 0.03 0.1 0.3 1 3 10 30]';
count = 1;
results = zeros((size(C_and_sigma, 1))^2, 3);
for C_t = 1:length(C_and_sigma),
    for sigma_t = 1:length(C_and_sigma),
        model= svmTrain(X, y, C_and_sigma(C_t), @(x1, x2) gaussianKernel(x1, x2, C_and_sigma(sigma_t))); 
        predictions = svmPredict(model, Xval);
        score_t = mean(double(predictions ~= yval));
        
        results(count, 1) = score_t;
        results(count, 2) = C_and_sigma(C_t);
        results(count, 3) = C_and_sigma(sigma_t);
        count = count + 1;
    end;   
end; 
fprintf('# Prediction error\tC\t\tSigma\n');
for i = 1:(size(C_and_sigma, 1))^2
    fprintf('  \t%f\t%f\t%f\n', results(i, 1), results(i, 2), results(i, 3));
end;
[min, id] = min(results(:, 1));
fprintf('Results BEST: %f\tC = %f\tsigma = %f\n', min,  results(id, 2), results(id, 3));
C =  results(id, 2);
sigma =  results(id, 3);
% =========================================================================

end
