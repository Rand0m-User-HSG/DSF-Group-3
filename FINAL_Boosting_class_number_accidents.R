##Maboost##

#####Introduction####

# In this script we perform a logistic regression for a classication task
# The goal is to predict the degree of severity of an accident
# There are 3 dgrees of severity: light injuries, severe injuries, fatalities
# We perdorm a multiclass classificaiton with help of boosting

rm(list =ls())

#maboost package is the extension of adaboost package for multiclass classification
#install.packages("maboost")
#install.packages("caret")
library(rpart)
library(C50)
library(maboost)
library(lattice)
library(tidyverse)
library(caret)


load("./Data/X_matrix_classification_number_accidents.RData")
load("./Data/Y_vector_classification_number_accidents.RData")



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
print(min(error_opt_cv))

##INSERER BEST PARAMETERS DANS LE MODEL WITH LEAVE ONE OUT

####Leave-one-out cross-validation###

k = nrow(X_matrix)
num_classes = 9
Empirical_error_LOO = rep(NA, k)
pred_LOO = matrix(NA, nrow = nrow(X_matrix)/k, ncol = k)
y_classified_LOO = rep(NA, length(Y_vector))
X_matrix = data.frame(X_matrix)

for (i in 1:k) {
  
  yn = Y_vector[-((1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k))]
  xn = X_matrix[-((1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)),]
  xn_test = data.frame(X_matrix[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k),])
  yn_test = Y_vector[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)]
  
  model_maboost_LOO = maboost(x = xn, y = yn, iter = 5, nu = .1, C50tree = T, C5.0Control(CF = .2, minCase = 128))
  
  pred_LOO = predict(model_maboost_LOO,xn_test,type="class");
  
  y_classified_LOO[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)] <- pred_LOO
  
  Empirical_error_LOO = length(which(y_classified_LOO != Y_vector)) / n
}

# Empirical_error_LOO =  ? 


####Misclassification matrix leave-one-out cross validation####
#The code is based on the script of JPO, Day 4Exercise4_Handwriting_recognition.R 

misclassification_matrix_LOO = matrix(0,num_classes, num_classes)

for (i in 1:num_classes) {
  for (j in 1:num_classes) {
    misclassification_matrix_LOO[i, j] = length(which((Y_vector == i) & (y_classified_LOO == j))) / length(which((Y_vector == i)))
  }
}

