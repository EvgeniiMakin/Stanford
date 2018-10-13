function [error_train, error_val] = learningCurvePro(X, y, Xval, yval, lambda, num_iter)
m = size(X, 1);
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m,
    error_t = 0;
    error_v = 0;
    for j = 1:num_iter,
          sel = randperm(m);
          sel = sel(1:i);
          theta = trainLinearReg(X(sel, :), y(sel, :), lambda);
          error_t = error_t + linearRegCostFunction(X(sel, :), y(sel, :), theta, 0);
          error_v = error_v + linearRegCostFunction(Xval, yval, theta, 0);
    end;
    error_train(i) = error_t/num_iter;
    error_val(i) = error_v/num_iter;
end;
end
