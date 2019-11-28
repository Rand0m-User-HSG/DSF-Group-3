--------------##10-folds cross-validation with optimization##-------------

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


load("./Data/covariate_matrix.RData")
load("./Data/Y_vector_classification.RData")

optim_boosting <- function(k, j, X, Y){
  
  fold = 10
  y_classified <- rep(NA, length(Y))
  cv_error <- rep(NA, fold)

  for (i in 1:fold) {
    
    yn = Y[-((1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold))]
    xn = X[-((1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold)),]
    xn_test = data.frame(X[(1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold),])
    yn_test = Y[(1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold)]
    
    model_maboost = maboost(x = xn, y = yn, iter = 1, verbose = 1 ,nu = .1, C50tree = T, C5.0Control(CF = j, minCase = k))
    
    pred = predict(model_maboost, xn_test, type="class")
    
    y_classified[(1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold)] <- pred
    
  }
  
  cv_error <- mean(abs(Y - y_classified))
  return(cv_error)
}

first = 7
last = 12
min_cases = rep(NA, last-first)
for (i in first:last){
  min_cases[i-first+1] <- 2^i
}
CF_possibilities <- seq(from = .1, to = 1, by = .01)
error <- matrix(NA, ncol = length(min_cases), nrow = length(CF_possibilities))
row = 0
col = 0
for (k in min_cases){
  col = col + 1
  for (j in CF_possibilities){
    row = row + 1
    error[row, col] = optim_boosting(k, j, X_matrix, Y_vector)
  }
}


best_parameters <- which(error == min(error), arr.ind = T)
print(best_parameters)
print(min(error))