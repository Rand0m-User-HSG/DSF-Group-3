###Introduction ###
#In this script we perform a boosting for a classification task
#We would like to classify the severity of an accident. 
#They are three degrees of severity: light injuries, severe injuries and fatalities. 

rm(list=ls())
library(tidyverse)
library(dplyr)

# we load the covariate matrix with dummy variables that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_classification.RData") # the name of this vector is Y_vector

---------------##First attempt with xgboost package##--------------------------
#Firstly, we use the xgbboost package to try performing this classification problem. 
library("xgboost")

#The code for this script is based on the exercise 14_Boosting.R in R scripts JP

xgb.fit1 <- xgb.cv(
  data = X_matrix,
  label = Y_vector,
  objective = "multi:softmax",
  num_class = 4,
  nrounds = 1000,
  nfold = 5,
  verbose = FALSE)


#xgboost has a bug with multiclass classification, this is why we need to use another R package.

-----------##Second attempt with gbm package##------------------------

#This part of the code is based on the book "An Introduction to Statistical Learning" pp. 330-331

library(gbm)

num_degrees = 3
p = dim(X_matrix)[2]
fold_10 = 10
beta = matrix(0,p + 1, num_degrees)
error = 0

for (i in 1:fold_10) {
  
  yn = Y_vector[(1+(i-1)*5429):(i*5429)]
  xn = X_matrix[(1+(i-1)*5429):(i*5429),]
  xn_test = X_matrix[-((1+(i-1)*5429):(i*5429)),]
  
  for (d in 1:num_degrees) {
    
    degree_selected=which(yn==d)
    y_d = yn
    y_d[-degree_selected] = 0  
    y_d[degree_selected] = 1
    
    data = data.frame(y_d,xn)
    data_test = data.frame(xn_test)
    
    model_gbmfit_d = gbm(y_d ~., data, distribution = "multinomial", n.trees = 100, interaction.depth = 4)
  
    pred <- predict.gbm(model_gbmfit_d, data_test, n.trees = 100, type="response")
    
    MSE[i] = (Y_vector[i] - pred[i])^2
  }
  Empirical_error_10_fold = MSE[i]
  error = error + Empirical_error_10_fold }

#I get the error message: 
#Erreur : objet 'MSE' introuvable
#De plus : Warning messages:
#1: In gbm.fit(x = x, y = y, offset = offset, distribution = distribution,  :  variable 11: AI has no variation.
#2: In gbm.fit(x = x, y = y, offset = offset, distribution = distribution,  : variable 12: AR has no variation.

#and the pred function does not give class but probability, weird stuff

----##Maboost##-----

#maboost package is the extension of adaboost package for multiclass classification
library(maboost)
library(caret)

num_degrees = 3
p = dim(X_matrix)[2]
fold_10 = 10
beta = matrix(0,p + 1, num_degrees)
error_cv = 0


for (i in 1:fold_10) {
  
  yn = Y_vector[(1+(i-1)*5429):(i*5429)]
  xn = X_matrix[(1+(i-1)*5429):(i*5429),]
  xn_test = X_matrix[-((1+(i-1)*5429):(i*5429)),]
  
  for (d in 1:num_degrees) {
    
    degree_selected=which(yn==d)
    y_d = yn
    y_d[-degree_selected] = 0  
    y_d[degree_selected] = 1
    
    data = data.frame(y_d,xn)
    data_test = data.frame(xn_test)
    
    model_maboostfit_d = maboost(y_d ~., data)
    
    pred.x= predict(model_maboostfit_d,data_test,type="class");
    
  }
  MSE[i] = ((Y_vector[i] - pred.x[i])^2)
  empirical_error = error + MSE[i]
}

