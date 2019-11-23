#### Introduction ####
# In this script we create a dataset specifically for the classification tasks
# that consist of predicting the number of accidents per day in canton ZH
# The idea is to create different classes, associated to different numbers of accidents
# We do so because the number of accidents is (1) small, and (2) discrete
# These two properties lend themselves well to classification-based predictions


rm(list=ls())
library(tidyverse)

# we load the Y_vector for regression tasks that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_regression.RData") # this data is loaded as Y_vector

# we load the X_matrix for regression tasks that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix_reg.RData") # this data is loaded as X_matrix


#### X_matrix ####

# we first create the X_matrix for classification tasks oriented towards the number of accidents
# in fact we just copy the existing X_matrix for regression tasks
# and save it under a different name

save(X_matrix, file = "Data/X_matrix_classification_number_accidents.RData")


#### Y_vector ####

# we observe that the values of Y_vector are contained in the range [0, 26]
min(Y_vector) # min(Y_vector) = 0
max (Y_vector) # max (Y_vector) = 26

# thus we divide this range into 9 classes of 3
for(i in 1:length(Y_vector)){
  if(Y_vector[i] >=0 & Y_vector[i] <= 2){
    Y_vector[i] = 1 # class 1 gets the value 1
  }
  if(Y_vector[i] >=3 & Y_vector[i] <= 5){
    Y_vector[i] = 2 
  }
  if(Y_vector[i] >=6 & Y_vector[i] <= 8){
    Y_vector[i] = 3 
  }
  if(Y_vector[i] >=9 & Y_vector[i] <= 11){
    Y_vector[i] = 4 
  }
  if(Y_vector[i] >=12 & Y_vector[i] <= 14){
    Y_vector[i] = 5 
  }
  if(Y_vector[i] >=15 & Y_vector[i] <= 17){
    Y_vector[i] = 6 
  }
  if(Y_vector[i] >=18 & Y_vector[i] <= 20){
    Y_vector[i] = 7 
  }
  if(Y_vector[i] >=21 & Y_vector[i] <= 23){
    Y_vector[i] = 8 
  }
  if(Y_vector[i] >=24 & Y_vector[i] <= 26){
    Y_vector[i] = 9 
  }
}

save(Y_vector, file = "Data/Y_vector_classification_number_accidents.RData")
