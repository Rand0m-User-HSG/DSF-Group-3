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
# and this code works (unlike in next section with cross-validation...)

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


#### Leave-one-out cross-validation ####

# in this section we build upon the results from the past section
# and endeavour to cross-validate these results
# however this fails, and we get the following error message:
# "Error in eval(family$initialize) : y values must be 0 <= y <= 1"
# this is particularly puzzling, because we make sure that y_d is between 0 and 1

# code for leave-one-out cross-validation is based on the script Exercise4_Functions.R
# JPO scripts -> day 2 -> cross-validation

num_degrees = 3
n = dim(X_matrix)[1]
p = dim(X_matrix)[2]
index= c(1:n)
beta_one_vs_all = matrix(0,p + 1, num_degrees)
error_cv = 0

for (i in 2:(n-1)) {
  
  indexn = index[-i]
  yn = Y_vector[indexn]
  xn = X_matrix[indexn, ]
  
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
  
  y_classified = apply(cbind(rep(1,(n-1)), xn) %*% beta_one_vs_all , 1, FUN=which.max)
  Empirical_error_one_vs_all = length(which(y_classified != yn)) / (n-1)
  error_cv = error_cv + Empirical_error_one_vs_all
  
}
