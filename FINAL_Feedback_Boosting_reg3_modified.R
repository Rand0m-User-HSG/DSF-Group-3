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



####Generalized boosted regression model without cross validation####
best_trees = 
best_interaction_depth = 2

#gbm requires data frames
X_matrix <- data.frame(X_matrix)

model_gbm = gbm(Y_vector  ~., data = X_matrix  ,distribution = "gaussian", 
                n.minobsinnode = 10,shrinkage = .01, n.trees = best_trees, interaction.depth = best_interaction_depth)

pred = predict.gbm(model_gbm, X_matrix ,n.trees = best_trees, type = "link")

yn_forecasted <- pred
MSE_regression = sum((Y_vector - yn_forecasted)^2, na.rm = T)  / (length(Y_vector))
MAE_regression = mean(abs(Y_vector - yn_forecasted), na.rm = T)

# MSE_regression = 6.60857
# MAE_regression = 2.013694

####10-folds cross-validation####
best_trees = 
best_interaction_depth = 2

fold = 10
sum_of_10_fold_cv_MSEs = 0
sum_of_10_fold_cv_MAEs = 0

yn_forecasted =  rep(NA, length(Y_vector)/fold)

for (i in 1:fold) {
  
  
  yn = Y_vector[-((1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold))]
  xn = data.frame(X_matrix[-((1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)),])
  xn_test = data.frame(X_matrix[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold),])
  yn_test = Y_vector[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)]
  
  model_gbm = gbm(yn  ~., data = xn ,distribution = "gaussian",n.minobsinnode = 10,shrinkage = .01, 
                  n.trees = best_trees, interaction.depth = best_interaction_depth)

  
  pred = predict.gbm(model_gbm, xn_test, n.trees = best_trees ,type = "link")
  
  yn_forecasted <- pred
  sum_of_10_fold_cv_MSEs = sum_of_10_fold_cv_MSEs + sum((yn_test - yn_forecasted)^2, na.rm = T)  / (length(yn_test))
  sum_of_10_fold_cv_MAEs = sum_of_10_fold_cv_MAEs + mean(abs(yn_test - yn_forecasted), na.rm = T)
}

MSE_10_fold_cv = sum_of_10_fold_cv_MSEs / fold
MAE_10_fold_cv = sum_of_10_fold_cv_MAEs / fold

# MSE_10_fold_cv = 
# MAE_10_fold_cv = 

# Comparison to the errors without cross-validation
# MSE_regression = 6.607747
# MAE_regression = 2.012888
# Getting slightly higher cv errors is normal, because they correspond to "testing" errors
# unlike the errors from the previous section which were merely "training" errors

####Leave-one-out cross-validation####
best_trees =
best_interaction_depth = 2

# Parallel computation
library(foreach)
library(doParallel)

X_matrix <- data.frame(X_matrix)

index=c(1:nrow(X_matrix))

cl = makeCluster(4)
#registerDoSNOW(cl)
registerDoParallel(cl)

start.time <- Sys.time()
se_cv = foreach (i =  1:nrow(X_matrix), .combine = rbind) %dopar% { 
  indexn = index[-i]
  
  yn = Y_vector[indexn]
  xn = X_matrix[indexn, ]
  
  library(gbm)
  
  fitresult_cv = gbm(yn  ~., data = xn  ,distribution = "gaussian", shrinkage = .01, 
                     n.minobsinnode = 10, interaction.depth = best_interaction_depth, n.trees = best_trees)
  
  (Y_vector[i] - predict.gbm(fitresult_cv, X_matrix[i, ], n.trees = best_trees, type = "link"))^2 
}

mse_cv_LOO = sum(se_cv) / nrow(X_matrix)

mae_cv_LOO = mean((se_cv)^0.5)

# mse_cv_LOO = 
# mae_cv_LOO = 

end.time <- Sys.time()
(time_parallel_computation = end.time - start.time)
stopCluster(cl)


####10-folds cross-validation with optimization####

optim_boosting_cv<- function(depth, trees, X, Y){
  
  fold = 10
  X <- data.frame(X)
  Y <- as.double(Y)
  
  model_gbm = gbm(Y ~., data = X, distribution = "gaussian", shrinkage = .01, 
                  n.trees = trees, n.minobsinnode = 10, interaction.depth = depth, cv.folds = 10)
  
  pred <- model_gbm$fit
  
  return(mean(abs(Y_vector - pred)))
}

optim_boosting_cv(2,3000, X_matrix, Y_vector)
# 2.196196
optim_boosting_cv(2, 4000, X_matrix, Y_vector)
# 2.16096
optim_boosting_cv(2, 5000, X_matrix, Y_vector)
# 2.130191
optim_boosting_cv(2, 6000, X_matrix, Y_vector)
# 2.102685
optim_boosting_cv(2, 7000, X_matrix, Y_vector)
# 2.079182
optim_boosting_cv(2, 8000, X_matrix, Y_vector)
# 2.057501
optim_boosting_cv(2, 9000, X_matrix, Y_vector)
# 2.03341
optim_boosting_cv(2, 10000, X_matrix,Y_vector)
#2.013958

possible_depth_opt_cv <- 1:2
possible_trees_opt_cv <- 9900:10100
#These values were optimzed because it was written on the help document that a shrinkage from 0.001 and 0.1
#with a n.trees combination from 3000 until 10000 values was optimal

row = 0 
col = 0
error_opt_cv <- matrix(NA, nrow = length(possible_trees_opt_cv), ncol = length(possible_depth_opt_cv))

for (fold in possible_depth_opt_cv){
  col = col +1
  row = 0
  for (j in possible_trees_opt_cv){
    row = row +1
    error_opt_cv[row, col] <- optim_boosting_cv(k, j, X_matrix, Y_vector)
  }
}

best_parameters_opt_cv<- which(error_opt_cv == min(error_opt_cv), arr.ind = T) # remember to adjust it based on the start of the possible_trees (and depth)
print(best_parameters_opt_cv)
#   
#interaction.depth = 2, n.trees = 
print(min(error_opt_cv))

#error_opt_cv = 
