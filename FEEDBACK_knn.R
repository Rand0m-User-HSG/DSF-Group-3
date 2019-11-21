##### KNN Classifier ####
# In this script we will apply KNN classification. 
# We want to classifiy the severity of accidents.
# Beforehand we cleaned the data into three categories: 
# Accident with light injuries (as3), Accident with severe injuries	(as2), Accident with fatalities	(as1).


rm(list=ls())
library(tidyverse)
library(dplyr)

# I've cut lines 12-13, because you had loaded the data for regression

# we load the covariate matrix with dummy variables that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_classification.RData") # the name of this vector is Y_vector





# We use this library for KNN.
library(class)


#### one-vs-all KNN classification ####

# in this section we perform a one-vs-all KNN classification
# without caring about cross-validation yet

# code for this section is based on the script Exercise2_ClassifierAnemia_2features.R
# JPO scripts, day 4, anemia classifier

Total_sample_x = X_matrix
n = nrow(X_matrix)

light_injuries_sample_x = as.data.frame(cbind(Y_vector, X_matrix)) %>%
  filter(Y_vector==3) %>%
  select(-Y_vector)
severe_injuries_sample_x = as.data.frame(cbind(Y_vector, X_matrix)) %>%
  filter(Y_vector==2) %>%
  select(-Y_vector)
fatalities_sample_x = as.data.frame(cbind(Y_vector, X_matrix)) %>%
  filter(Y_vector==1) %>% 
  select(-Y_vector)

# create the Y vectors for one-vs-all classification
Y_light_injuries = rep(0, length(Y_vector))
Y_severe_injuries = rep(0, length(Y_vector))
Y_fatalities = rep(0, length(Y_vector))
for(i in 1:length(Y_vector)){
  Y_light_injuries[i] = ifelse(Y_vector[i] == 3, 1, 0)
  Y_severe_injuries[i] = ifelse(Y_vector[i] == 2, 1, 0)
  Y_fatalities[i] = ifelse(Y_vector[i] == 1, 1, 0)
}

# Training errors
labels_light_injuries = knn(Total_sample_x, light_injuries_sample_x, Y_light_injuries, k = 5,prob=TRUE)
labels_severe_injuries = knn(Total_sample_x, severe_injuries_sample_x, Y_severe_injuries, k = 5,prob=TRUE)
labels_fatalities = knn(Total_sample_x, fatalities_sample_x, Y_fatalities, k = 5, prob = TRUE)
# attr(labels_***,"prob") contains the probability to appear in the winning class

TypeI_errors_KNN_light_injuries = sum(as.numeric(as.character(labels_light_injuries)))
TypeII_errors_KNN_light_injuries = length(which(as.numeric(as.character(labels_light_injuries)) == 0))
TypeI_errors_KNN_severe_injuries = sum(as.numeric(as.character(labels_severe_injuries)))
TypeII_errors_KNN_severe_injuries = length(which(as.numeric(as.character(labels_severe_injuries)) == 0))
TypeI_errors_KNN_fatalities = sum(as.numeric(as.character(labels_fatalities)))
TypeII_errors_KNN_fatalities = length(which(as.numeric(as.character(labels_fatalities)) == 0))

Empirical_error_KNN = (TypeI_errors_KNN_light_injuries + TypeII_errors_KNN_light_injuries + TypeI_errors_KNN_severe_injuries + TypeII_errors_KNN_severe_injuries + TypeI_errors_KNN_fatalities + TypeII_errors_KNN_fatalities) / (3*n)

# Empirical_error_KNN = 0.3333333


#### 10-fold cross-validation ####


fold = 10

TypeI_errors_KNN_light_injuries_cv = rep(0, fold)
TypeII_errors_KNN_light_injuries_cv = rep(0, fold)
TypeI_errors_KNN_severe_injuries_cv = rep(0, fold)
TypeII_errors_KNN_severe_injuries_cv = rep(0, fold)
TypeI_errors_KNN_fatalities_cv = rep(0, fold)
TypeII_errors_KNN_fatalities_cv = rep(0, fold)

labels_light_injuries_cv = rep(0, fold)
labels_severe_injuries_cv = rep(0, fold)
labels_fatalities_cv = rep(0, fold)

for(i in 1:fold){
  
  lower_bound_i = (i-1)*(round(nrow(X_matrix)/fold))+1
  upper_bound_i = round(nrow(X_matrix)/fold*i)
  
  total_sample_x_i = Total_sample_x[(lower_bound_i:upper_bound_i), ]
  light_injuries_sample_x_i = light_injuries_sample_x[(lower_bound_i:upper_bound_i), ]
  severe_injuries_sample_x_i = severe_injuries_sample_x[(lower_bound_i:upper_bound_i), ]
  fatalities_sample_x_i = fatalities_sample_x[(lower_bound_i:upper_bound_i), ]
  Y_light_injuries_i = Y_light_injuries[lower_bound_i:upper_bound_i]
  Y_severe_injuries_i = Y_severe_injuries[lower_bound_i:upper_bound_i]
  Y_fatalities_i = Y_fatalities[lower_bound_i:upper_bound_i]
  # the X_matrices and Y_vectors of fold i (i.e. 1/10 of the corresponding X_matrices and Y_vectors)
  
  labels_light_injuries_cv[i] = knn(total_sample_x_i, light_injuries_sample_x_i, Y_light_injuries_i, k = 5,prob=TRUE)
  labels_severe_injuries_cv[i] = knn(total_sample_x_i, severe_injuries_sample_x_i, Y_severe_injuries_i, k = 5,prob=TRUE)
  labels_fatalities_cv[i] = knn(total_sample_x_i, fatalities_sample_x_i, Y_fatalities_i, k = 5, prob = TRUE)
  
  TypeI_errors_KNN_light_injuries_cv[i] = sum(as.numeric(as.character(labels_light_injuries_cv[i])))
  TypeII_errors_KNN_light_injuries_cv[i] = length(which(as.numeric(as.character(labels_light_injuries_cv[i])) == 0))
  TypeI_errors_KNN_severe_injuries_cv[i] = sum(as.numeric(as.character(labels_severe_injuries_cv[i])))
  TypeII_errors_KNN_severe_injuries_cv[i] = length(which(as.numeric(as.character(labels_severe_injuries_cv[i])) == 0))
  TypeI_errors_KNN_fatalities_cv[i] = sum(as.numeric(as.character(labels_fatalities_cv[i])))
  TypeII_errors_KNN_fatalities_cv[i] = length(which(as.numeric(as.character(labels_fatalities_cv[i])) == 0))
  
}

Empirical_error_KNN_cv = (mean(TypeI_errors_KNN_light_injuries_cv) + mean(TypeII_errors_KNN_light_injuries_cv) + mean(TypeI_errors_KNN_severe_injuries_cv) + mean(TypeII_errors_KNN_severe_injuries_cv) + mean(TypeI_errors_KNN_fatalities_cv) + mean(TypeII_errors_KNN_fatalities_cv)) / (3*n)
