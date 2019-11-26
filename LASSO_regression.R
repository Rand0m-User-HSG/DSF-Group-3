####### LASSO regression #######



load("~/Desktop/HSG/DSF/Group Work/Data/covariate_matrix.RData")
load("~/Desktop/HSG/DSF/Group Work/Data/Y_vector_regression.RData")


library(tidyverse)
library(glmnet)       # implementing regularized regression approaches



# The code is based on the script of JPO, Day 1, Polynomial regression splines Ex_2_2_polynomial_regression_parallel.txt
# install.packages("doParallel")
library(doParallel)

# Parallel computation
cl = makeCluster(4)
#registerDoSNOW(cl)
registerDoParallel(cl)

mod_cv <- cv.glmnet(x=X_matrix, y=Y_vector,   nfolds = nrows(X_matrix), family='multinomial')

# Find the best lambda
mod_cv$lambda.1se

