--------------##10-folds cross-validation with optimization##-------------

rm(list = ls())

library(gbm)

load("./Data/covariate_matrix_reg.RData")
load("./Data/Y_vector_regression.RData")

optim_boosting<- function(trees, X, Y){
  
  fold = 10
  X <- data.frame(X)
  Y <- as.double((Y))
  
  model_gbm = gbm(Y ~., data = X, distribution = "gaussian", shrinkage = .1, n.trees = trees, 
                    interaction.depth = 2, cv.folds = fold)
  
  pred <- model_gbm$fit
  
  return(mean(abs(Y_vector - pred)))
}

possible_trees <- 100:101
error <- rep(NA, length(possible_trees))

for (j in possible_trees){
  error[j-99] <- optim_boosting(j, X_matrix, Y_vector)
}

best_trees <- 99 + which(error == min(error))
print(best_trees)
print(min(error))