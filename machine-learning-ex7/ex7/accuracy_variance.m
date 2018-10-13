function accuracy = accuracy_variance(S, K)
S_diag = S * ones(size(S,1),1);
accuracy = (sum(S_diag(1:K,1))/sum(S_diag))*100;
end