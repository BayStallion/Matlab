function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

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

Cval = [ 0.01, 0.03, 0.10, 0.30, 1.00, 3.00, 10.0, 30.0];
sigmaval = Cval;
best_cost = 10000000;

for i = 1:length(Cval)
    for j = 1:length(sigmaval)
        model = svmTrain(X, y, Cval(i), ...
            @(x1, x2) gaussianKernel(x1, x2, sigmaval(j)), 0.00001, 5);
        
        predictions = svmPredict(model, Xval);
        
        new_cost = mean(double(predictions ~= yval));
        
        if (new_cost < best_cost)
            best_cost = new_cost;
            C_best = Cval(i);
            sigma_best = sigmaval(j);
        end
    end
end

C = C_best;
sigma = sigma_best;

% =========================================================================

end
