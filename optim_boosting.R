--------------##10-folds cross-validation with optimization##-------------

rm(list = ls())

library(gbm)

load("./Data/covariate_matrix_reg.RData")
load("./Data/Y_vector_regression.RData")

optim_boosting<- function(depth, trees, X, Y){
  
  fold = 10
  X <- data.frame(X)
  Y <- as.double(Y)
  
  model_gbm = gbm(Y ~., data = X, distribution = "gaussian", shrinkage = .1, n.trees = trees, 
                    interaction.depth = depth, cv.folds = fold)
  
  pred <- model_gbm$fit
  
  return(mean(abs(Y_vector - pred)))
}

possible_depth <- 1:2
possible_trees <- 100:101

row = 0 
col = 0
error <- matrix(NA, nrow = length(possible_trees), ncol = length(possible_depth)

for (k in possible_depth){
  col = col +1
  for (j in possible_trees){
    row = row +1
    error[row, col] <- optim_boosting(k, j, X_matrix, Y_vector)
  }
}

best_parameters <- which(error == min(error), arr.ind = T) # remember to adjust it based on the start of the possible_trees (and depth)
print(best_parameters)
print(min(error))
