#### Introduction ####
# In this script we perform a logistic regression for a classication task
# The goal is to predict the degree of severity of an accident
# There are 3 degrees of severity: light injuries, severe injuries, fatalities
# We thus perform a so-called one-vs-all logistic regression
# We validate our results by performing a 10-fold cross-validation
# And visualise these results with a misclassification matrix


rm(list=ls())
library(tidyverse)

# we load the covariate matrix with dummy variables that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_classification.RData") # the name of this vector is Y_vector


#### one-vs-all logistic regression ####

# in this section we perform a simple one-vs-all logistic regression
# without caring about cross-validation yet

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

# Empirical_error_one_vs_all = 0.2163198

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

num_degrees = 3
p = dim(X_matrix)[2]
fold = 10
beta_one_vs_all_cv = matrix(0,p + 1, num_degrees)
y_classified_cv = rep(0, length(Y_vector))
sum_of_10_fold_cv_errors = 0

for (i in 1:fold) {
  
  lower_bound_i = (i-1)*(round(nrow(X_matrix)/fold))+1
  upper_bound_i = round(nrow(X_matrix)/fold*i)
  
  x_i = X_matrix[(lower_bound_i:upper_bound_i), ]
  y_i = Y_vector[lower_bound_i:upper_bound_i]
  # the X_matrix and Y_vector of fold i (i.e. 1/10 of X_matrix and Y_vector)
  
  x_non_i = X_matrix[-(lower_bound_i:upper_bound_i), ]
  y_non_i = Y_vector[-(lower_bound_i:upper_bound_i)]
  # the X_matrix and Y_vector of everything that is not fold i (i.e. 9/10 of X_matrix and Y_vector)  
  
  for (d in 1:num_degrees) {
    
    degree_selected=which(y_non_i==d)
    y_d = y_non_i
    y_d[-degree_selected] = 0  
    y_d[degree_selected] = 1
    
    data = data.frame(y_d,x_non_i)
    model_glmfit_d = glm(y_d ~., data, start =rep(0,p+1) ,family=binomial(link="logit"),
                         control=list(maxit = 100, trace = FALSE) )
    beta_glmfit_d  = model_glmfit_d$coefficients
    beta_glmfit_d[is.na(beta_glmfit_d)]=0
    
    beta_one_vs_all_cv[, d] = beta_glmfit_d 
  }
  
  y_classified_cv[lower_bound_i:upper_bound_i] = apply(cbind(rep(1,round(nrow(X_matrix)/fold)), x_i) %*% beta_one_vs_all_cv , 1, FUN=which.max)
  empirical_error_for_fold_i = length(which(y_classified_cv[lower_bound_i:upper_bound_i] != y_i)) / (round(nrow(X_matrix)/fold))
  sum_of_10_fold_cv_errors = sum_of_10_fold_cv_errors + empirical_error_for_fold_i
  # we use x_non_i and y_non_i to calculate beta, and test this beta on fold i
}

error_10_fold_cv = sum_of_10_fold_cv_errors/fold

# error_10_fold_cv = 0.2164119 (Empirical_error_one_vs_all = 0.2163198)

# Getting a slightly higher cv error is normal, because it corresponds to a "testing" error
# unlike the error from the previous section which was merely a "training" error


misclassification_matrix_cv = matrix(0,num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_cv[i, j] = length(which((Y_vector == i) & (y_classified_cv == j))) / length(which((Y_vector == i)))
  }
}
