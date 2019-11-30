rm(list=ls())
library(tidyverse)

# we load the covariate matrix with dummy variables that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_classification.RData") # the name of this vector is Y_vector


#### Introduction ####
# In this script we train and evaluate different models for a classication task.
# The goal is to predict the degree of severity of an accident.
# There are 3 degrees of severity: light injuries, severe injuries, fatalities.
# We're gonna use a logistic regression, a knn, a boosting-model, and a neural network.
# We validate our results by performing a 10-fold cross-validation,
# and visualise these results with a misclassification matrix

#### We start with logistic regression ####

# 10-fold cross-validation

# in this section we build upon the results from the past section
# and endeavour to cross-validate these results
# we use 10-fold cross-validation to save computational time

num_degrees = length(unique(Y_vector))
p = ncol(X_matrix)
fold = 10
beta_one_vs_all_cv = matrix(0, p + 1, num_degrees)
y_classified_log = rep(0, length(Y_vector))
sum_of_10_fold_cv_errors = 0

for (i in 1:fold) {
  
  lower_bound_i = (i-1)*(round(nrow(X_matrix)/fold))+1
  upper_bound_i = i*nrow(X_matrix)/fold
  
  x_test = X_matrix[lower_bound_i:upper_bound_i, ]
  y_test = Y_vector[lower_bound_i:upper_bound_i]

  x_train = X_matrix[-(lower_bound_i:upper_bound_i), ]
  y_train = Y_vector[-(lower_bound_i:upper_bound_i)]
  # the X_matrix and Y_vector of everything that is not fold i (i.e. 9/10 of X_matrix and Y_vector)  
  
  for (d in 1:num_degrees) {
    
    degree_selected = which(y_train == d)
    y_d = y_train
    y_d[-degree_selected] = 0  
    y_d[degree_selected] = 1
    
    data = data.frame(y_d, x_train)
    model_glmfit_d = glm(y_d ~., data, start =rep(0,p+1), family=binomial(link="logit"),
                         control=list(maxit = 100, trace = FALSE) )
    beta_glmfit_d  = model_glmfit_d$coefficients
    beta_glmfit_d[is.na(beta_glmfit_d)]=0
    
    beta_one_vs_all_cv[, d] = beta_glmfit_d 
  }
  
  y_classified_log[lower_bound_i:upper_bound_i] = apply(cbind(rep(1,round(nrow(X_matrix)/fold)), x_test) %*% beta_one_vs_all_cv , 1, FUN=which.max)
}

cv_error_log = mean(abs(Y_vector - y_classified_log))
print(cv_error_log)

misclassification_matrix_log = matrix(0, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_log[i, j] = length(which((Y_vector == i) & (y_classified_log == j))) / length(which((Y_vector == i)))
  }
}
print(misclassification_matrix_log)


#### we continue with the KNN ####

# We use this library for KNN.

# install.packages("class")
library(class)

# since knn takes an hyperparameter (that is k, the number of neighbours to consider),
# we optimized over it.
# For choosing the range over which to optimize, we first started with the rule of thumb that k = sqrt(nrow(X_matrix)),
# then we took an intervall of +/- sqrt(k) to get our optimization range.
# The k which resulted in the smallest error was k = ***.
# As it takes quite some time to run the optimization, feel free to skip this part after running the following:
# k_best = 233

optim_knn <- function(K, X, Y){
  fold <- 10
  cv_error <- rep(NA, fold)
  
  for (i in 1:fold){
    
    X_k <- X[-((1 + (i-1)*nrow(X)/fold):(i*(nrow(X)/fold))),]
    Y_k <- Y[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X)/fold)))]
    X_test <- X[(1 + (i-1)*nrow(X)/fold) : (i*(nrow(X)/fold)),]
    Y_test <- Y[(1 + (i-1)*nrow(X)/fold) : (i*(nrow(X)/fold))]
    
    pred <- as.numeric(knn(X_k, X_test, cl = Y_k, k = K))
    
    cv_error[i] <- mean(abs(Y_test - pred))
    
  }
  
  errors <- mean(cv_error)
  
  return(errors)
}

possible_k <- 210:250
error <- rep(NA, length(possible_k))
index = 0

for (i in possible_k){
  index = index + 1
  error[index] <- optim_knn(i, X_matrix, Y_vector)
}

k_best <- which(error == min(error)) + 210 -1
print(k_best)
cv_error_knn = min(error)
print(cv_error_knn)

# We now run a 10-fold cv with the best k we found in the optimization

fold <- 10
cv_error_knn <- rep(NA, fold)
y_classified_knn <- rep(NA, nrow(X_matrix))

for (i in 1:fold){
  
  X_k <- X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold))),]
  Y_k <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold)))]
  X_test <- X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)),]
  Y_test <- Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))]
  
  pred <- as.numeric(knn(X_k, X_test, cl = Y_k, k = 9))
  
  y_classified_knn[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))] <- pred
  cv_error_knn[i] <- mean(abs(Y_test - pred))
  
}

cv_error_knn <- mean(cv_error_knn)
print(cv_error_knn)  # we get an error of .2362498

misclassification_matrix_knn = matrix(0, unique(Y_vector), unique(Y_vector))
for (i in 1:unique(Y_vector)) {
  for (j in 1:unique(Y_vector)) {
    misclassification_matrix_knn[i ,j] = length(which((Y_vector == i) & (y_classified_knn == j))) / length(which((Y_vector == i)))
  }
}
print(misclassification_matrix_knn)



#### Let's procede with boosting ####

#install.packages("rpart")
#install.packages("c50")
#install.packages("lattice")
#install.packages("tidyverse")
#install.packages("maboost")
#install.packages("caret")
library(rpart)
library(C50)
library(maboost)
library(lattice)
library(tidyverse)
library(caret)

# 10-folds cross-validation with optimization


optim_boosting <- function(cf, mincases, X, Y){
  
  fold = 10
  error <- rep(NA, fold)
  
  for (i in 1:fold) {
    
    y_train = Y[-((1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold))]
    x_train = X[-((1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold)),]
    x_test = data.frame(X[(1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold),])
    y_test = Y[(1+(i-1)*nrow(X)/fold):(i*nrow(X)/fold)]
    
    model = maboost(x = x_train, y = y_train, iter = 1, verbose = 1 ,nu = .1, C50tree = T, C5.0Control(CF = cf, minCase = mincases))
    
    pred = predict(model, x_test, type="class")
    
    error[i] <- mean(abs(y_test - as.double(pred)))
    
  }
  
  return(mean(error))
}

min_cases = 2:6

CF_possibilities <- seq(from = .1, to = 1, by = .1)

errors <- matrix(NA, ncol = length(min_cases), nrow = length(CF_possibilities))

col = 0
for (k in min_cases){
  col = col + 1
  row = 0
  for (j in CF_possibilities){
    row = row + 1
    errors[row, col] = optim_boosting(k, j, X_matrix, Y_vector)
  }
}

best_parameters_boosting <- which(errors == min(errors), arr.ind = T)
print(best_parameters_boosting)
best_cases <- best_parameters_boosting[2] + 2 - 1
print(best_cases)
best_CF <- best_parameters_boosting[1]/10
print(best_CF)
print(min(errors))

# let's run a quick 10-fold cv with the best parameters
# If you skipped the optimization please run this 2 lines:
# best_cases <- 4
# best_CF <- .5

fold = 10
y_classified_boosting <- rep(NA, length(Y_vector))
error_boosting <- rep(NA, fold)

for (i in 1:fold) {
  
  y_train = Y_vector[-((1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold))]
  x_train = X_matrix[-((1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)),]
  x_test = data.frame(X_matrix[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold),])
  y_test = Y_vector[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)]
  
  model = maboost(x = x_train, y = y_train, iter = 1, verbose = 1 ,nu = .1, C50tree = T, C5.0Control(CF = best_CF, minCase = best_cases))
  
  pred = predict(model, x_test, type="class")
  
  y_classified_boosting[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)] <- pred
  error_boosting[i] <- mean(abs(y_test - as.double(pred)))
  
}

misclassification_matrix_boosting = matrix(0, unique(Y_vector), unique(Y_vector))

for (i in 1:unique(Y_vector)) {
  for (j in 1:unique(Y_vector)) {
    misclassification_matrix_boosting[i ,j] = length(which((Y_vector == i) & (y_classified_boosting == j))) / length(which((Y_vector == i)))
  }
}

print(misclassification_matrix_boosting)
cv_error_boosting <- mean(error_boosting)
print(cv_error_boosting) # cv_error_boosting = 0.2251428


#### last but not least we train a neural network ####

# DISCLAIMER: In order to train these models, you'll need to be able to use keras 
# in a python IDE (this means you also need TensorFlow), that is, 
# you also need to (pip) install these packages in python!
# Here is a link to a page in which they explain how to do it: 
# https://towardsdatascience.com/setup-an-environment-for-machine-learning-and-deep-learning-with-anaconda-in-windows-5d7134a3db10
# However, nearing the end of these script, we trained a CNN and saved the weights, so that
# even if you aren't able to run the models to get the predictions and the errors, we can get them with 
# highschool-level linear algebra!

#install.packages("tensor")
#install.packages("keras")
#install.packages("tensorflow")
#install.packages("Rcpp")
#install.packages("rlist")
library(rlist)
library(tensor)
library(tensorflow)
library(keras)
library(tidyverse)
#install_tensorflow()
train <- tensorflow::train  # the library "caret" masks the function train of the library tensorflow, this is a bit rudimental, but works

Y_categorized <- to_categorical(Y_vector - 1)  # keras requires a specific kind of vector as target, at the same time the indexing in python starts from 0

build_model <- function(){
  
  model <- keras_model_sequential() %>%
    layer_dense(units = ncol(X_matrix), activation = "sigmoid",
                input_shape = ncol(X_matrix)) %>%
    layer_dense(units = 128, activation = "sigmoid") %>%
    layer_dense(units = ncol(Y_categorized), activation = "softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(),
    metrics = list("accuracy")
  )
  model
  
}

# we start by optimizing over the number of epochs, the reason is that we want to avoid using a validation split

optim_NN <- function(e, X, Y){
  
  fold = 10
  cv_error <- rep(NA, fold)
  
  for (i in 1:fold){
    
    X_k <- X[-((1 + (i-1)*nrow(X)/fold) : (i*nrow(X)/fold)),]
    Y_k <- Y[-((1 + (i-1)*nrow(X)/fold) : (i*nrow(X)/fold)),]
    X_test <- X[(1 + (i-1)*nrow(X)/fold) : (i*nrow(X)/fold),]
    Y_test <- Y[(1 + (i-1)*nrow(X)/fold) : (i*nrow(X)/fold),]
    
    model <- build_model()
    
    training <- model %>% fit(
      X_k, Y_k, 
      epochs = e, batch_size = 128,
      validation.split = 0,
      verbose =  0)
    
    pred <- model %>% predict(X_test)
    
    cv_error[i] <- mean(abs(Y_test - pred))
    
  }
  
  return(mean(cv_error))
}

# since we're training the model with a validation.split of 0, we risk overfitting it, so we search for the epochs,
# which get the smallest MAE on unseen data

possible_eopchs = 1:15
errors <- rep(NA, length(possible_eopchs))
for (i in possible_eopchs){
  errors[i] <- optim_NN(i, X_matrix, Y_categorized)   # Depending on your machine you may see a warning, it doesn't matter for us
}

best_epochs <- which(errors == min(errors))           
print(best_epochs)
cv_error_NN <- mean(errors)                    # this is our best MAE in a 10-fold CV, the epochs were 4 and the MAE .1704369
print(cv_error_NN)                             # we used predict instead of predict_classes to keep the probabilities, so the next MAE will be bigger


# Let's use the best number of epochs to build a non-cv model that won't overfit
# The reason for this is to save the weights, so that if you can't run keras you will still see some result
# best_epochs = 4

model <- build_model()

training <- model %>% fit(
  X_matrix, Y_categorized, 
  epochs = best_epochs, batch_size = 128,
  validation.split = 0,
  verbose =  1)

y_classified_NN<- 1 + model %>% predict_classes(X_matrix)
error <- mean(abs(Y_vector - y_classified_NN))
print(error)

misclassification_matrix_NN = matrix(0, unique(Y_vector), unique(Y_vector))
for (i in 1:unique(Y_vector)) {
  for (j in 1:unique(Y_vector)) {
    misclassification_matrix_NN[i ,j] = length(which((Y_vector == i) & (y_classified_NN == j))) / length(which((Y_vector == i)))
  }
}
print(misclassification_matrix_NN)

weights_reg <- model$get_weights()
list.save(weights_reg, "Data/weights_reg.RData")

# here we use the weights of the model to predict with matrix multiplication

load("./Data/weights_reg.RData")
Beta <- x

Beta_input <- Beta[[1]]
Beta_input2 <- Beta[[2]]
Beta_hidden <- Beta[[3]]
Beta_hidden2 <- Beta[[4]]
Beta_output <- Beta[[5]]
Beta_output2 <- Beta[[6]]

sigmoid = function(z) {
  x = 1/(1 + exp(z))
}

softmax = function(z){
  x = exp(z) / sum(exp(z))
}

X <- cbind(rep(1, nrow(X_matrix)), X_matrix)
Beta_input_true <- cbind(Beta_input2, t(Beta_input))

Input_layer <- Beta_input_true %*% sigmoid(t(X))

Input_layer <- t(Input_layer)
Input_layer<- cbind(rep(1, nrow(Input_layer)), Input_layer)
Beta_hidden_true <- cbind(Beta_hidden2, t(Beta_hidden))

Hidden_layer <- Beta_hidden_true %*% sigmoid(t(Input_layer))

Hidden_layer <- t(Hidden_layer)
Hidden_layer <- cbind(rep(1, nrow(Hidden_layer)), Hidden_layer)
Beta_output_true <- cbind(Beta_output2, t(Beta_output))

Pred <- Beta_output_true %*% softmax(t(Hidden_layer))
Pred <- t(Pred)
Pred <- apply(Pred , 1, FUN=which.max)

error_NN = mean(abs(Y_vector - Pred))
print(error_NN)                                                           # the error stays the same



#### We now quickly compare the results ####

# The various errors:
print(paste("The error of the logistic regression:", cv_error_log))
print(paste("The error of the KNN:", cv_error_knn))
print(paste("The error of boosting:", cv_error_boosting))
print(paste("The error of the neural network:", cv_error_NN))

# The various misclassification matrices:
print("The misclassificaiton matrix of the logistic regression:")
print(misclassification_matrix_log)
print("The misclassificaiton matrix of the KNN:")
print(misclassification_matrix_knn)
print("The misclassificaiton matrix of boosting:")
print(misclassification_matrix_boosting)
print("The misclassificaiton matrix of the neural network:")
print(misclassification_matrix_NN)

# As expected the NN performs better, followed close by boosting.
# That said our models coudn't find a "real" way to predict the severity of the accident,
# instead they all started predicting every accident to be a light injury, that is a class number 3.
# The reasons for this can be various:
# Firstly our data is heavily biased on the light injuries, which take around 78.58% of the data points,
# while severe_injuries take 20.35% and fatalities only 1.07%. If this is a problem we tried 2 solutions in the
# scripts sampling.R and SMOTE.R, in which we work with less biased data.
# Another reasons could be that our data is uncorrect, especially the wheater data. We think we can strike this reason out
# as it's not believable from a respected site like the NOAA to have corrupted data. We also used meteoswiss for the data
# about the number of accidents and we still coudn't really predcit anything.
# The last and more probable reason is that our data is too noisy and/or there's no enough correlation between
# wheater and timeframe of an accident and its severity.
