##Maboost##

#####Introduction####

# In this script we perform a logistic regression for a classication task
# The goal is to predict the degree of severity of an accident
# There are 3 dgrees of severity: light injuries, severe injuries, fatalities
# We perdorm a multiclass classificaiton with help of boosting

rm(list =ls())

#install.packages("maboost")
#install.packages("caret")
#install.packages("C50")

library(rpart)
library(C50)
library(maboost)
library(lattice)
library(tidyverse)
library(caret)

load("./Data/covariate_matrix.RData")
load("./Data/Y_vector_classification.RData")

####10-folds cross-validation with optimization####

optim_boosting_opt_cv <- function(k, j, X, Y){
  
  fold = 10
  y_classified_opt_cv <- rep(NA, length(Y))
  error_opt_cv <- rep(NA, fold)
  
  for (i in 1:fold) {
    
    yn = Y[-((1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold))]
    xn = X[-((1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold)),]
    xn_test = data.frame(X[(1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold),])
    yn_test = Y[(1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold)]
    
    model_maboost_opt_cv = maboost(x = xn, y = yn, iter = 1, verbose = 1 ,nu = .1, C50tree = T, C5.0Control(CF = j, minCase = k))
    
    pred_opt_cv = predict(model_maboost_opt_cv, xn_test, type="class")
    
    y_classified_opt_cv[(1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold)] <- pred_opt_cv
    
  }
  
  error_opt_cv <- mean(abs(Y - y_classified_opt_cv))
  return(error_opt_cv)
}

min_cases_opt_cv = 2:6

CF_possibilities_opt_cv <- seq(from = .1, to = 1, by = .1)

error_opt_cv <- matrix(NA, ncol = length(min_cases_opt_cv), nrow = length(CF_possibilities_opt_cv))

col = 0
for (k in min_cases_opt_cv){
  col = col + 1
  row = 0
  for (j in CF_possibilities_opt_cv){
    row = row + 1
    error_opt_cv[row, col] = optim_boosting_opt_cv(k, j, X_matrix, Y_vector)
  }
}
best_parameters_opt_cv <- which(error_opt_cv == min(error_opt_cv), arr.ind = T)
print(best_parameters_opt_cv)
# 3  5 
print(min(error_opt_cv))
# 0.2248665

####Misclassification matrix 10-folds cross validation####
#The code is based on the script of JPO, Day 4Exercise4_Handwriting_recognition.R 

misclassification_matrix_opt_cv = matrix(0, num_degrees, num_degrees)

for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_opt_cv[i, j] = length((which(Y_vector == i == i) & (y_classified_opt_cv == j))) / length(which((Y_vector == i == i)))
  }
}
