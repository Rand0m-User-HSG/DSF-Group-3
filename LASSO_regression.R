####### LASSO regression #######

rm(list = ls())
load("./Data/covariate_matrix_reg.RData")
load("./Data/Y_vector_regression.RData")


library(tidyverse)
library(glmnet)       # implementing regularized regression approaches



# The folowing code is based on the script of JPO, Day 1, Polynomial regression splines Ex_2_2_polynomial_regression_parallel.txt
# install.packages("doParallel")
library(iterators)
library(parallel)
library(doParallel)

# Parallel computation
cl = makeCluster(32)

registerDoParallel(cl)

#The following code runs a LASSO regression on our data. It belongs to the glmnet library.
mod_cv <- cv.glmnet(x=X_matrix, y=Y_vector, nfolds = nrow(X_matrix), family = "gaussian", alpha = 1)

stopCluster(cl)

# Find the best lambda
mod_cv$lambda.1se



#Now we want the betas
beta=coef(mod_cv)
print(beta)
