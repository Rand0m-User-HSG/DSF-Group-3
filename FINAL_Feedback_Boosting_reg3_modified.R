##GBM##

#####Introduction####

# In this script we perform a generalized a boosted regresion model 
# The goal is to predict the number of accidents in canton ZH on a particular day
# We test our results by performing 10-fold cross-validation

rm(list =ls())

# install.packages("gbm")
# install.packages("caret")
# install.packages("doParallel")

library(tidyverse)
library(gbm)
library(caret)
library(doParallel)


# we load the covariate matrix with dummy variables that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix_reg.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_regression.RData") # the name of this vector is Y_vector


# The code for Parallel computation is based on the script of JPO, Day 1, Polynomial regression splines Ex_2_2_polynomial_regression_parallel.txt

# Parallel computation
cl = makeCluster(6)
#registerDoSNOW(cl)
registerDoParallel(cl)

####Generalized boosted regression model without cross validation####

#gbm requires data frames
X_matrix <- data.frame(X_matrix)

model_gbm = gbm(Y_vector  ~., data = X_matrix  ,distribution = "gaussian",cv.folds = 10,n.minobsinnode = 10,shrinkage = .01, n.trees = 3000)

pred = predict.gbm(model_gbm, X_matrix ,n.trees = 3000, type = "link")

yn_forecasted <- pred
MSE_regression = sum((Y_vector - yn_forecasted)^2, na.rm = T)  / (length(Y_vector))
MAE_regression = mean(abs(Y_vector - yn_forecasted), na.rm = T)

# MSE_regression = 8.754159
# MAE_regression = 2.308763

####10-folds cross-validation####


k = 10
sum_of_10_fold_cv_MSEs = 0
sum_of_10_fold_cv_MAEs = 0

yn_forecasted =  rep(NA, length(Y_vector)/k)

for (i in 1:k) {
  
  
  yn = Y_vector[-((1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k))]
  xn = data.frame(X_matrix[-((1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)),])
  xn_test = data.frame(X_matrix[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k),])
  yn_test = Y_vector[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)]
  
  model_gbm = gbm(yn  ~., data = xn  ,distribution = "gaussian",n.minobsinnode = 10,shrinkage = .01, n.trees = 3000)
  
  pred = predict.gbm(model_gbm, xn_test, n.trees = 3000 ,type = "link")
  
  yn_forecasted <- pred
  sum_of_10_fold_cv_MSEs = sum_of_10_fold_cv_MSEs + sum((yn_test - yn_forecasted)^2, na.rm = T)  / (length(yn_test))
  sum_of_10_fold_cv_MAEs = sum_of_10_fold_cv_MAEs + mean(abs(yn_test - yn_forecasted), na.rm = T)
}

MSE_10_fold_cv = sum_of_10_fold_cv_MSEs / k
MAE_10_fold_cv = sum_of_10_fold_cv_MAEs / k

# MSE_10_fold_cv = 9.395641
# MAE_10_fold_cv = 2.391612

# Comparison to the errors without cross-validation
# MSE_regression = 8.754159
# MAE_regression = 2.308763
# Getting slightly higher cv errors is normal, because they correspond to "testing" errors
# unlike the errors from the previous section which were merely "training" errors

####Leave-one-out cross-validation####


# Parallel computation
library(foreach)
library(doParallel)

index=c(1:nrow(X_matrix))

cl = makeCluster(4)
#registerDoSNOW(cl)
registerDoParallel(cl)

start.time <- Sys.time()
sse_cv = foreach (i =  1:nrow(X_matrix), .combine = rbind) %dopar% { 
  indexn = index[-i]
  
  yn = Y_vector[indexn]
  xn = X_matrix[indexn, ]
  
  library(gbm)
  
  fitresult_cv = gbm(yn  ~., data = xn  ,distribution = "gaussian", shrinkage = .01, 
                     n.minobsinnode = 10, n.trees = best_trees)
  
  (Y_vector[i] - predict.gbm(fitresult_cv, X-matrix[i, ], n.trees = best_trees, type = "link"))^2 
}
mse_cv_LOO = sum(se_cv) / nrow(X_matrix)

#mse_cv_LOO = ?

end.time <- Sys.time()
(time_parallel_computation = end.time - start.time)
stopCluster(cl)


####10-folds cross-validation with optimization####

optim_boosting_cv<- function(depth, trees, X, Y){
  
  fold = 10
  X <- data.frame(X)
  Y <- as.double(Y)
  
  model_gbm = gbm(Y ~., data = X, distribution = "gaussian", shrinkage = .01, 
                  n.trees = trees, n.minobsinnode = 10, interaction.depth = depth)
  
  pred <- model_gbm$fit
  
  return(mean(abs(Y_vector - pred)))
}

optim_boosting_cv(3000, X_matrix, Y_vector)
optim_boosting_cv(4000, X_matrix, Y_vector)
optim_boosting_cv(5000, X_matrix, Y_vector)
optim_boosting_cv(6000, X_matrix, Y_vector)
optim_boosting_cv(7000, X_matrix, Y_vector)
optim_boosting_cv(8000, X_matrix, Y_vector)
optim_boosting_cv(9000, X_matrix, Y_vector)
optim_boosting_cv(10000, X_matrix, Y_vector)

possible_depth_opt_cv <- 1:2
possible_trees_opt_cv <- 9750:10000

row = 0 
col = 0
error_opt_cv <- matrix(NA, nrow = length(possible_trees_opt_cv), ncol = length(possible_depth_opt_cv))

for (k in possible_depth_opt_cv){
  col = col +1
  for (j in possible_trees_opt_cv){
    row = row +1
    error_opt_cv[row, col] <- optim_boosting_cv(k, j, X_matrix, Y_vector)
  }
}

best_parameters_opt_cv<- which(error_opt_cv == min(error_opt_cv), arr.ind = T) # remember to adjust it based on the start of the possible_trees (and depth)
print(best_parameters_opt_cv)
print(min(error_opt_cv))

#error_opt_cv = ?

####Leave-one-out cross-validation with optimization##-##

optim_boosting_LOO<- function(depth, trees, X, Y){
  
  fold = nrow(X_matrix)
  X <- data.frame(X)
  Y <- as.double(Y)
  
  model_gbm = gbm(Y ~., data = X, distribution = "gaussian", shrinkage = .01, 
                  n.trees = trees, n.minobsinnode = 10, interaction.depth = depth)
  
  pred <- model_gbm$fit
  
  return(mean(abs(Y_vector - pred)))
}

possible_depth_opt_LOO<- 1:2
possible_trees_opt_LOO <- 9750:10000

row = 0 
col = 0
error_opt_LOO <- matrix(NA, nrow = length(possible_trees_opt_LOO), ncol = length(possible_depth_opt_LOO))

for (k in possible_depth_opt_LOO){
  col = col +1
  for (j in possible_trees_opt_LOO){
    row = row +1
    error_opt_LOO[row, col] <- optim_boosting_LOO(k, j, X_matrix, Y_vector)
  }
}

best_parameters_opt_LOO<- which(error_opt_LOO == min(error_opt_LOO), arr.ind = T) # remember to adjust it based on the start of the possible_trees (and depth)
print(best_parameters_opt_LOO)
print(min(error_opt_LOO))

#error_opt_LOO = ?

