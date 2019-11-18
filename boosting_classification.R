###Introduction ###

#In this script we perform a boosting for a classification task
#We would like to classify the severity of an accident. 
#They are three degrees of severity: light injuries, severe injuries and fatalities. 

rm(list=ls())
library(tidyverse)
library(dplyr)


---------------##First attempt with xgboost package##--------------------------
#Firstly, we use the xgbboost package to try performing this classification problem. 

library(xgboost)

# we load the covariate matrix with dummy variables that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_classification.RData") # the name of this vector is Y_vector

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

#gbm requires a data frame and not a matrix
X_df <- data.frame(X_matrix)
Y_df <- data.frame(X_matrix)

#Randomly shuffle the data
X_df = X_df[sample(nrow(X_df)),]
Y_vector = Y_df[sample(nrow(Y_df)),]

##-------Cross valation with k = 10 folds ------------------------
#I try to perform the model with this one and when I get it I will try to do it with leave-one-out
#cross-validation. 

#Create 10 equally size folds
folds = cut(seq(1,nrow(X_df)),breaks=10,labels=FALSE)
folds_y = cut(seq(1,nrow (Y_df)),breaks=10,labels=FALSE)

#Perform 10 folds cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- X_df[testIndexes, ]
  trainData <- X_df[-testIndexes, ]
}

for(i in 1:10){
  #Segement your data by fold using the which() function
  Yindexes <- which(folds_y==i,arr.ind=TRUE)
  Ytest<- Y_df[Yindexes, ]
  Ytrain <- Y_df[-Yindexes, ]
}

model_fit_train <- gbm(Ytrain ~., data = trainData, distribution = "multinomial", 
                 n.trees = 100, interaction.depth = 4)

#here is the new error message I get: Error in model.frame.default(formula = Ytrain ~ ., 
#data = trainData, drop.unused.levels = TRUE,  : 
#type (list) incorrect pour la variable 'Ytrain'


#If I try to do it for X_matrix, without separating the training and testing data, I get
#and it runs forever without giving any output. 

model_fit <- gbm(Y_vector~., data = X_df, distribution = "multinomial", 
                       n.trees = 100, interaction.depth = 4)
