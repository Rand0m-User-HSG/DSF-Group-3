
##GBM##

--------------##Introduction##-------------

# In this script we perform a generalized a boosted regresion model 
# The goal is to predict the number of accidents in canton ZH on a particular day
# We test our results by performing 10-fold cross-validation

rm(list =ls())

# install.packages("gbm")
# install.packages("caret")

library(tidyverse)
library(gbm)
library(caret)

# I've cut lines 19 to 22 (not necessary)
#
#
#

# we load the covariate matrix with dummy variables that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix_reg.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_regression.RData") # the name of this vector is Y_vector


--------------##Generalized boosted regression model##-------------
#gbm requires data frames

# I've cut lines 31 to 32 (not necessary)
#
X_matrix <- data.frame(X_matrix)

model_gbm = gbm(Y_vector  ~., data = X_matrix  ,distribution = "gaussian",cv.folds = 10,
                shrinkage = .01, n.minobsinnode = 10,n.trees = 500)

pred = predict.gbm(model_gbm, X_matrix ,type = "link")

yn_forecasted <- pred
MSE = sum((Y_vector - yn_forecasted)^2, na.rm = T)  / (length(Y_vector)) # I've renamed it into MSE because the use of sum in the formulation was confusing
MAE = mean(abs(Y_vector - yn_forecasted), na.rm = T) # I've renamed it into MAE because the use of sum in the formulation was confusing

# MSE = 9.980673 I get slightly different results (before: MSE  = 9.987382)
# MAE = 2.47481 I get slightly different results (before: MAE = 2.47698)


--------------##10-folds cross-validation##-------------

k = 10
sum_of_10_fold_cv_MSEs = 0 # changed the variable name, because it must be the same as in line 74
sum_of_10_fold_cv_MAEs = 0 # changed the variable name, because it must be the same as in line 75

# I've cut line 57 (not necessary)
yn_forecasted =  rep(NA, length(Y_vector)/k) # changed the length of yn_forecasted (divided it by k)

for (i in 1:k) {
  
  
  yn = Y_vector[-((1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k))]
  xn = data.frame(X_matrix[-((1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)),])
  xn_test = data.frame(X_matrix[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k),])
  yn_test = Y_vector[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)]
  
  model_gbm = gbm(yn  ~., data = xn  ,distribution = "gaussian",cv.folds = 10,
                  shrinkage = .01, n.minobsinnode = 10,n.trees = 500)
  
  pred = predict.gbm(model_gbm, xn_test ,type = "link")
  
  yn_forecasted <- pred
  sum_of_10_fold_cv_MSEs = sum_of_10_fold_cv_MSEs + sum((yn_test - yn_forecasted)^2, na.rm = T)  / (length(yn_test)) # replaced Y_vector by yn_test, because it must have the same length as yn_forecasted; also adjusted the formulation of the error variable (see line 57)
  sum_of_10_fold_cv_MAEs = sum_of_10_fold_cv_MAEs + mean(abs(yn_test - yn_forecasted), na.rm = T)
  # we use xn_test and yn_test to do the prediction on fold i
}

# I've cut lines 79-80 (no need to specify this, as this is only an intermediate value)
# 

MSE_10_fold_cv = sum_of_10_fold_cv_MSEs / k
MAE_10_fold_cv = sum_of_10_fold_cv_MAEs / k

# MSE_10_fold_cv = 10.26497 (before: MSE_10_fold_cv = 1.024888)
# MAE_10_fold_cv = 2.509592 (before: MAE_10_fold_cv = 0.2511079)

# Comparison to the errors without cross-validation
# MSE = 9.980673 (my results, i.e. from the feedback)
# MAE = 2.47481 (my results, i.e. from the feedback)

# Getting slightly higher cv errors is normal, because they correspond to "testing" errors
# unlike the errors from the previous section which were merely "training" errors
