#### Introduction ####
# In this script we perform a logistic regression for a classication task
# The goal is to predict the degree of severity of an accident
# There are 3 dgrees of severity: light injuries, severe injuries, fatalities
# We thus perform a so-called one-vs-all logistic regression


rm(list=ls())
library(tidyverse)

# we load the covariate matrix with dummy variables that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_classification.RData") # the name of this vector is Y_vector


#### one-vs-all logistic regression ####

# in this section we perform a simple one-vs-all logistic regression
# without caring about cross-validation

# code for this section is based on the script Exercise4_Handwriting_recognition.R (question 4)
# JPO scripts -> day 4 -> digit recognition

num_degrees = 3
n = dim(X_matrix)[1]
p = dim(X_matrix)[2]
beta_one_vs_all = matrix(0,p + 1, num_degrees)

for (d in 1:num_degrees) {
  
  degree_selected=which(Y_vector==d)
  y_d = Y_vector
  y_d[-degree_selected] = 0  
  y_d[degree_selected] = 1
  
  data = data.frame(y_d,X_matrix)
  model_glmfit_d = glm(y_d ~., data, start =rep(0,p+1) ,family=binomial(link="logit"),
                       control=list(maxit = 100, trace = FALSE) )
  beta_glmfit_d  = model_glmfit_d$coefficients
  beta_glmfit_d[is.na(beta_glmfit_d)]=0
  
  beta_one_vs_all[, d] = beta_glmfit_d 
}

y_classified = apply(cbind(rep(1,n), X_matrix) %*% beta_one_vs_all , 1, FUN=which.max)
Empirical_error_one_vs_all = length(which(y_classified != Y_vector)) / n

misclassification_matrix = matrix(0,num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix[i, j] = length(which((Y_vector == i) & (y_classified == j))) / length(which((Y_vector == i)))
  }
}


#### 10-fold cross-validation ####

# in this section we build upon the results from the past section
# and endeavour to cross-validate these results
# we use 10-fold cross-validation to save computational time

# code for cross-validation is based on the script Exercise4_Functions.R
# JPO scripts -> day 2 -> cross-validation

num_degrees = 3
p = dim(X_matrix)[2]
fold_10 = 10
beta_one_vs_all = matrix(0,p + 1, num_degrees)
error_cv = 0

for (i in 1:fold_10) {
  
  yn = Y_vector[(1+(i-1)*5429):(i*5429)]
  xn = X_matrix[(1+(i-1)*5429):(i*5429), ]
  
  for (d in 1:num_degrees) {
    
    degree_selected=which(yn==d)
    y_d = yn
    y_d[-degree_selected] = 0  
    y_d[degree_selected] = 1
    
    data = data.frame(y_d,xn)
    model_glmfit_d = glm(y_d ~., data, start =rep(0,p+1) ,family=binomial(link="logit"),
                         control=list(maxit = 100, trace = FALSE) )
    beta_glmfit_d  = model_glmfit_d$coefficients
    beta_glmfit_d[is.na(beta_glmfit_d)]=0
    
    beta_one_vs_all[, d] = beta_glmfit_d 
  }
  
  y_classified = apply(cbind(rep(1,5429), xn) %*% beta_one_vs_all , 1, FUN=which.max)
  Empirical_error_10_fold = length(which(y_classified != yn)) / 5429
  error_cv = error_cv + Empirical_error_10_fold
  
}

error_10_fold_cv = error_cv/10

# the error with 10-fold cv is slightly better than without 10-fold cv
# Empirical_error_one_vs_all = 0.2163198
# error_10_fold_cv = 0.2149383


#### 89-fold cross-validation ####

# in this section we build upon the results from the past section
# and endeavour to cross-validate these results
# we use 89-fold cross-validation, hoping to get a lower error
# indeed 54290 (total number of observations) can be divided by 89 into an integer (610)

# code for cross-validation is based on the script Exercise4_Functions.R
# JPO scripts -> day 2 -> cross-validation

num_degrees = 3
p = dim(X_matrix)[2]
fold_89 = 89
beta_one_vs_all = matrix(0,p + 1, num_degrees)
error_cv = 0

for (i in 1:fold_89) {
  
  yn = Y_vector[(1+(i-1)*610):(i*610)]
  xn = X_matrix[(1+(i-1)*610):(i*610), ]
  
  for (d in 1:num_degrees) {
    
    degree_selected=which(yn==d)
    y_d = yn
    y_d[-degree_selected] = 0  
    y_d[degree_selected] = 1
    
    data = data.frame(y_d,xn)
    model_glmfit_d = glm(y_d ~., data, start =rep(0,p+1) ,family=binomial(link="logit"),
                         control=list(maxit = 100, trace = FALSE) )
    beta_glmfit_d  = model_glmfit_d$coefficients
    beta_glmfit_d[is.na(beta_glmfit_d)]=0
    
    beta_one_vs_all[, d] = beta_glmfit_d 
  }
  
  y_classified = apply(cbind(rep(1,610), xn) %*% beta_one_vs_all , 1, FUN=which.max)
  Empirical_error_89_fold = length(which(y_classified != yn)) / 610
  error_cv = error_cv + Empirical_error_89_fold
  
}

error_89_fold_cv = error_cv/89

# the error with 89-fold cv is slightly better than with 10-fold cv and without cv at all
# Empirical_error_one_vs_all = 0.2163198
# error_10_fold_cv = 0.2149383
# error_89_fold_cv = 0.1966661


#### trial 10-fold with different error calculation ####

# in this section we use a different error calculation
# each of the 10 beta matrices is used to calculate y_classified over the entire dataset
# we thus produce 10 different errors
# and pick the lowest one

# code for cross-validation is based on the script Exercise4_Functions.R
# JPO scripts -> day 2 -> cross-validation

num_degrees = 3
p = dim(X_matrix)[2]
fold_10 = 10
beta_one_vs_all = matrix(0,p + 1, num_degrees*fold_10)
Empirical_error_10_fold = rep(0, fold_10) # we will store the 10 errors in this vector

for (i in 1:fold_10) {
  
  yn = Y_vector[(1+(i-1)*5429):(i*5429)]
  xn = X_matrix[(1+(i-1)*5429):(i*5429), ]
  
  for (d in 1:num_degrees) {
    
    degree_selected=which(yn==d)
    y_d = yn
    y_d[-degree_selected] = 0  
    y_d[degree_selected] = 1
    
    data = data.frame(y_d,xn)
    model_glmfit_d = glm(y_d ~., data, start =rep(0,p+1) ,family=binomial(link="logit"),
                         control=list(maxit = 100, trace = FALSE) )
    beta_glmfit_d  = model_glmfit_d$coefficients
    beta_glmfit_d[is.na(beta_glmfit_d)]=0
    
    beta_one_vs_all[, d*i] = beta_glmfit_d 
  }
  
  y_classified = apply(cbind(rep(1,nrow(X_matrix)), X_matrix) %*% beta_one_vs_all[, (1+(i-1)*3):(i*3)] , 1, FUN=which.max)
  Empirical_error_10_fold[i] = length(which(y_classified != Y_vector)) / nrow(X_matrix)
}

error_10_fold_cv = min(Empirical_error_10_fold) # we pick the lowest error from the 10

# the error with 10-fold cv is slightly worse than without 10-fold cv, and worse than with the first version of 10-fold cv
# Empirical_error_one_vs_all = 0.2163198
# error_10_fold_cv = 0.2219377
