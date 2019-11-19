#### Introduction ####

# In this script we perform a linear regression
# The goal is to predict the number of accidents in canton ZH on a particular day
# We test our results by performing 10-fold cross-validation


rm(list=ls())
library(tidyverse)


# we load the covariate matrix with dummy variables that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix_reg.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_regression.RData") # the name of this vector is Y_vector


#### linear regression ####

# in this section we perform a simple linear regression
# without caring about cross-validation

# we first build the beta-vector
beta = solve((t(X_matrix) %*% X_matrix), (t(X_matrix)%*%Y_vector))

# and then we compute the associated MSE and MAE
Y_forecasted = X_matrix %*% beta
MSE_regression  = sum((Y_vector - Y_forecasted)^2)  / (length(Y_vector))
MAE_regression = mean(abs(Y_vector - Y_forecasted))

# MSE_regression  = 9.362053
# MAE_regression = 2.376562


#### 10-fold cross-validation ####

# in this section we perform a linear regression
# and carry out a 10-fold cross-validation

fold = 10
sum_of_10_fold_cv_MSEs = 0
sum_of_10_fold_cv_MAEs = 0

for (i in 1:fold){
  
  lower_bound_i = (i-1)*(round(nrow(X_matrix)/fold))+1
  upper_bound_i = round(nrow(X_matrix)/fold*i)
  
  x_i = X_matrix[(lower_bound_i:upper_bound_i), ]
  y_i = Y_vector[lower_bound_i:upper_bound_i]
  # the X_matrix and Y_vector of fold i (i.e. 1/10 of X_matrix and Y_vector)
  
  x_non_i = X_matrix[-(lower_bound_i:upper_bound_i), ]
  y_non_i = Y_vector[-(lower_bound_i:upper_bound_i)]
  # the X_matrix and Y_vector of everything that is not fold i (i.e. 9/10 of X_matrix and Y_vector)
  
  beta = solve((t(x_non_i) %*% x_non_i), (t(x_non_i)%*%y_non_i))
  
  y_i_forecasted = x_i %*% beta
  sum_of_10_fold_cv_MSEs = sum_of_10_fold_cv_MSEs + sum((y_i - y_i_forecasted)^2)  / (length(y_i))
  sum_of_10_fold_cv_MAEs = sum_of_10_fold_cv_MAEs + mean(abs(y_i - y_i_forecasted))
  # we use x_non_i and y_non_i to calculate beta, and test this beta on fold i
  
}

MSE_10_fold_cv = sum_of_10_fold_cv_MSEs / fold
MAE_10_fold_cv = sum_of_10_fold_cv_MAEs / fold

# MSE_10_fold_cv = 9.671956 (MSE_regression  = 9.362053)
# MAE_10_fold_cv = 2.413695 (MAE_regression = 2.376562)
