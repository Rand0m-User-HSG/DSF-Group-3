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

##Confusion matrix 1
# this is for the last loop only
num_degrees = 3
# we have to add this line, otherwise R doesn't know which value to take because there were initially three values: 1,2,3 and 
# by default, it just take the first one and will then stop at 1 and not continue
misclassification_matrix = matrix(NA, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix[i, j] = length(which((Y_vector == i) & (y_classified == j))) / length(which((Y_vector == i)))
  }
}
# the ouput is a bit strange. 

##Confusion matrix 2: a bit complicated but the ouput gives interesting info (even if it was not was I wanted as output) 
misclassification_matrix = matrix(NA, num_degrees, num_degrees)

for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix = confusionMatrix(
      factor(y_classified, levels = 1:3),
      factor(Y_vector, levels = 1:3), positive = NULL,
      dnn = c("Prediction", "Reference"), prevalence = NULL,
       mode = "sens_spec")
  }
}
misclassification_matrix = as.matrix(misclassification_matrix, what = "classes")
print(misclassification_matrix, 
      mode = x$mode, digits = max(3, getOption("digits") - 3), printStats = TRUE)
