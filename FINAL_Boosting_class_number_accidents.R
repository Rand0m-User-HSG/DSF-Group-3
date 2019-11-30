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

load("./Data/X_matrix_classification_number_accidents.RData")
load("./Data/Y_vector_classification_number_accidents.RData")


# 10-folds cross-validation with optimization


optim_boosting <- function(cf, mincases, X, Y){
  
  fold = 10
  error <- rep(NA, fold)
  
  for (i in 1:fold) {
    
    y_train = Y[-((1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold))]
    x_train = X[-((1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold)),]
    x_test = data.frame(X[(1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold),])
    y_test = Y[(1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold)]
    
    model = maboost(x = x_test, y = y_test, iter = 1, verbose = 1 ,nu = .1, C50tree = T, C5.0Control(CF = j, minCase = k))
    
    pred = predict(model, x_test, type="class")
    
    error[i] <- mean(abs(y_test - pred_opt_cv))
    
  }
  
  return(error)
}

min_cases = 2:6

CF_possibilities <- seq(from = .1, to = 1, by = .1)

errors <- matrix(NA, ncol = length(min_cases_opt_cv), nrow = length(CF_possibilities_opt_cv))

col = 0
for (k in min_cases){
  col = col + 1
  row = 0
  for (j in CF_possibilities){
    row = row + 1
    errors[row, col] = optim_boosting_opt_cv(k, j, X_matrix, Y_vector)
  }
}

best_parameters_boosting <- which(errors == min(errors), arr.ind = T)
print(best_parameters_boosting)
best_cases <- best_parameters_boosting[2] + 2 - 1
print(best_cases)
best_CF <- best_parameters_boosting[1]/10
print(best_CF)
print(min(errors))


##PLEASE PLUG THE BEST VALUES IN THE THREE FOLLOWING MODELS
####Generalized multicass classification without cross validation####

num_classes = 9
n = dim(X_matrix)[1]
p = dim(X_matrix)[2]
pred = matrix(NA, nrow = nrow(X_matrix), ncol = 1)
y_classified = rep(NA, length(Y_vector))
X_matrix = data.frame(X_matrix)

model_maboost = maboost(X_matrix, Y_vector, iter = 5, nu = .1, C50tree = T, C5.0Control(CF = .2, minCase = 128))

pred = predict(model_maboost,X_matrix,type="class")

y_classified =  pred

Empirical_error = length(which(y_classified != Y_vector)) / n

# Empirical_error = ?


####10-folds cross-validation####

fold = 10
num_classes = 9
Empirical_error_cv = rep(NA, fold)
pred_cv = matrix(NA, nrow = nrow(X_matrix)/fold, ncol = fold)
y_classified_cv = rep(NA, length(Y_vector))
X_matrix = data.frame(X_matrix)

for (i in 1:fold) {
  
  yn = Y_vector[-((1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold))]
  xn = X_matrix[-((1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)),]
  xn_test = data.frame(X_matrix[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold),])
  yn_test = Y_vector[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)]
  
  model_maboost_cv = maboost(x = xn, y = yn, iter = 5, nu = .1, C50tree = T, C5.0Control(CF = .2, minCase = 128))
  
  pred_cv = predict(model_maboost_cv,xn_test,type="class");
  
  y_classified_cv[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)] <- pred_cv
  
  Empirical_error_cv = length(which(y_classified_cv != Y_vector)) / n
}

# Empirical_error_cv = ?

####Misclassification matrix 10-folds cross validation####
#The code is based on the script of JPO, Day 4Exercise4_Handwriting_recognition.R 

misclassification_matrix_cv = matrix(0,num_classes, num_classes)

for (i in 1:num_classes) {
  for (j in 1:num_classes) {
    misclassification_matrix_cv[i, j] = length(which((yn == i) & (y_classified_cv == j))) / length(which((yn == i)))
  }
}

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
  
  Empirical_error_LOO[i] = length(which(y_classified_LOO[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)] != yn_test)) / length(yn_test)
}

# Empirical_error_LOO =  ? 

