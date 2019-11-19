----##Maboost##-----

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

k = 10
num_degrees <- unique(Y_vector)
error_cv = rep(NA, k)
pred= matrix(NA, nrow = nrow(X_matrix)/k, ncol = k)
y_classified <- rep(NA, length(Y_vector))

for (i in 1:k) {
  
  yn = Y_vector[-((1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k))]
  xn = X_matrix[-((1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)),]
  xn_test = data.frame(X_matrix[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k),])
  yn_test = Y_vector[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)]
    
  model_maboost = maboost(x = xn, y = yn, iter = 5, nu = .1, C50tree = T, C5.0Control(CF = .2, minCase = 128))
  
  pred= predict(model_maboost,xn_test,type="class");

  y_classified[(1+(i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)] <- pred
  error_cv[i] = mean(abs(yn_test - as.integer(pred)))
}

# this is for the last loop only
misclassification_matrix = matrix(NA, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix[i, j] = length(which((Y_vector == i) & (y_classified == j))) / length(which((Y_vector == i)))
  }
}
