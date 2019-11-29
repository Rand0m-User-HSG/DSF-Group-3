##### KNN Classifier ####
# In this script we will apply KNN classification. 
# We want to classifiy the amount of accidents happening on one day.
# Beforehand we cleaned the data into three categories: 

rm(list=ls())
library(tidyverse)


# we load the covariate matrix with dummy variables that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix_reg.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_regression.RData") # the name of this vector is Y_vector

# We use this library for KNN.
library(class)

########################################################## optimization

optim_knn <- function(K, X, Y){
  fold <- 10
  cv_error <- rep(NA, fold)
  
  for (i in 1:fold){
    
    X_k <- X[-((1 + (i-1)*nrow(X)/fold):(i*(nrow(X)/fold))),]
    Y_k <- Y[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X)/fold)))]
    X_test <- X[(1 + (i-1)*nrow(X)/fold) : (i*(nrow(X)/fold)),]
    Y_test <- Y[(1 + (i-1)*nrow(X)/fold) : (i*(nrow(X)/fold))]
    
    pred <- as.numeric(knn(X_k, X_test, cl = Y_k, k = K))
    
    cv_error[i] <- mean(abs(Y_test - pred))
    
  }
  
  errors <- mean(cv_error)
  
  return(errors)
}

# We use the interval of 1:60, as there is a rule of thumb, that the best k is 
# approx. the sqrt of the number of observations, which in this case is 53.8.
possible_k <- 40:60
error <- rep(NA, length(possible_k))
for (i in possible_k){
  error[i] <- optim_knn(i, X_matrix, Y_vector)
}

k_best <- which(error == min(error))
print(k_best) #53
# p <- optim(1, optim_knn, X = X_matrix, Y = Y_vector,  method = "Brent", lower = 1, upper = 2)  # the parameters are real values, while we only want integers

############################################################################ 10 fold cv with best k

fold <- 10
cv_error <- rep(NA, fold)
y_classified <- rep(NA, nrow(X_matrix))

for (i in 1:fold){
  
  X_k <- X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold))),]
  Y_k <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold)))]
  X_test <- X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)),]
  Y_test <- Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))]
  
  pred <- as.numeric(knn(X_k, X_test, cl = Y_k, k = k_best))
  
  y_classified[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)))] <- pred
  cv_error[i] <- mean(abs(Y_test - pred))
  
}

print(mean(cv_error))

misclassification_matrix = matrix(0, unique(Y_vector), unique(Y_vector))
for (i in 1:unique(Y_vector)) {
  for (j in 1:unique(Y_vector)) {
    misclassification_matrix[j ,i] = length(which((Y_vector == j) & (y_classified == i))) / length(which((Y_vector == j)))
  }
}
print(misclassification_matrix)
