function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for i=1:K,
    sum_t = zeros(1, n);
    number_for_this_K = 0;
    for j=1:m,
        if idx(j) == i,
            number_for_this_K = number_for_this_K + 1;
            sum_t = sum_t + X(j,:);
        end; 
    end;
    if number_for_this_K == 0,
        centroids(i,:) = zeros(1, n);
    else,
        centroids(i,:) = sum_t./number_for_this_K;
    endif;
end;






% =============================================================


end

