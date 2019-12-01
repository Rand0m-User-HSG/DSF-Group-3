rm(list=ls())
library(tidyverse)

# we load the covariate matrix with dummy variables that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix_reg.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_regression.RData") # the name of this vector is Y_vector

# we start by approximating to 19 every number of accidents bigger than it

sum(Y_vector > 19)/length(Y_vector) # 0.35%
Y_vector[which(Y_vector > 19)] <- 19
Y_vector <- Y_vector + 1  # this helps simplify the code, we just need to remember to take 1 out from the predictions

#### Introduction ####
# In this script we train and evaluate different models for a classication task.
# The goal is to predict the number of accidents happening in a given day in the canton Zürich.
# After approximating any number of accidents bigger than 19 to 19, we still have 20 classes.
# We're gonna use a logistic regression, a knn, a boosting-model, and a neural network.
# We validate our results by performing a 10-fold cross-validation,
# and visualise these results with a misclassification matrix.

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

cv_error_log = length(which(Y_vector != y_classified_log))/length(Y_vector)
print(cv_error_log) # cv_error_log = 0.8726269

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

# DISCLAIMER: we noticed that with the knn() function there's a bit of randomness involved, which even by setting a seed 
# doesn't disappear, therefore you can expect slightly different results than ours

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
sqrt(nrow(X_matrix)) # 54
sqrt(sqrt(nrow(X_matrix))) # 7.3

# we take one sqrt() to the left and one to the rigth, so
possible_k <- 46:62
error <- rep(NA, length(possible_k))
index = 0

for (i in possible_k){
  index = index + 1
  error[index] <- optim_knn(i, X_matrix, Y_vector)
}

k_best <- which(error == min(error)) + 46 -1
print(k_best)  # the best k is unrelevant, as the errors are extremely similars, for good form we're gonna pick 54
cv_error_knn = min(error)
print(cv_error_knn)
print(error)   # the best k is unrelevant, as the errors are extremely similars, for good form we're gonna pick 54
# We now run a 10-fold cv with the best k we found in the optimization

k_best = 54
fold <- 10
cv_error_knn <- rep(NA, fold)
y_classified_knn <- rep(NA, nrow(X_matrix))

for (i in 1:fold){
  
  X_k <- X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold))),]
  Y_k <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold)))]
  X_test <- X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)),]
  Y_test <- Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))]
  
  pred <- as.numeric(knn(X_k, X_test, cl = Y_k, k = k_best))
  
  y_classified_knn[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))] <- pred
  cv_error_knn[i] <- length(which(Y_test != pred))/length(Y_test)
  
}

cv_error_knn <- mean(cv_error_knn)
print(cv_error_knn) # cv_error_knn = 0.8795848

misclassification_matrix_knn = matrix(0, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
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
  
  return(mean(abs(error)))
}

min_cases = 2:102

CF_possibilities <- seq(from = .1, to = 1, by = .01)

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
print(best_parameters_boosting) # row=26 and col=68
best_cases <- best_parameters_boosting[2] + 2 - 1
print(best_cases) # best_cases = 69
best_CF <- best_parameters_boosting[1]/10
print(best_CF) # best_CF = 2.6
print(min(errors)) # min(errors) = 2.992042
print(errors)  # the errors don't seem to get better or worse with different parameters, so we just pick those best one
# let's run a quick 10-fold cv with the best parameters
# If you skipped the optimization please run this 2 lines:
# best_cases <- 4
# best_CF <- .5

fold = 10
y_classified_boosting <- rep(NA, length(Y_vector))
cv_error_boosting <- rep(NA, fold)

for (i in 1:fold) {
  
  y_train = Y_vector[-((1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold))]
  x_train = X_matrix[-((1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)),]
  x_test = data.frame(X_matrix[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold),])
  y_test = Y_vector[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)]
  
  model = maboost(x = x_train, y = y_train, iter = 1, verbose = 1 ,nu = .1, C50tree = T, C5.0Control(CF = best_CF, minCase = best_cases))
  
  pred = predict(model, x_test, type="class")
  
  y_classified_boosting[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)] <- pred
  cv_error_boosting[i] <- length(which(y_test != as.double(pred)))/length(y_test)
  
}

cv_error_boosting <- mean(error_boosting)
print(cv_error_boosting)
misclassification_matrix_boosting = matrix(0, num_degrees, num_degrees)

for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_boosting[i ,j] = length(which((Y_vector == i) & (y_classified_boosting == j))) / length(which((Y_vector == i)))
  }
}

print(misclassification_matrix_boosting)


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
    layer_dense(units = ncol(Y_categorized), activation = "sigmoid")
  
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
    
    pred <- model %>% predict_classes(X_test)
    
    cv_error[i] <- length(which(Y_vector[(1 + (i-1)*nrow(X)/fold) : (i*nrow(X)/fold)] != pred))
    
  }
  
  return(sum(cv_error)/length(Y_vector))
}

# since we're training the model with a validation.split of 0, we risk overfitting it, so we search for the epochs,
# which get the smallest MAE on unseen data

possible_eopchs = 1:25
errors <- rep(NA, length(possible_eopchs))
for (i in possible_eopchs){
  errors[i] <- optim_NN(i, X_matrix, Y_categorized)   # Depending on your machine you may see a warning, it doesn't matter for us
}

best_epochs <- which(errors == min(errors))           
print(best_epochs)
print(min(errors))                    # this is our best MAE in a 10-fold CV, the epochs were 15 and the errors 0.8864342
print(errors)

# Let's use the best number of epochs to build a non-cv model that won't overfit
# The reason for this is to save the weights, so that if you can't run keras you will still see some result
# best_epochs = 8

model <- build_model()

training <- model %>% fit(
  X_matrix, Y_categorized, 
  epochs = best_epochs, batch_size = 128,
  validation.split = 0,
  verbose =  1)

y_classified_NN<- model %>% predict_classes(X_matrix)
cv_error_NN <- length(which(Y_vector != y_classified_NN))/length(Y_vector)
print(cv_error_NN)          # we get a way smaller error than before (still substantial though), because we don't take the means twice

misclassification_matrix_NN = matrix(0, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
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

Pred <- Beta_output_true %*% sigmoid(t(Hidden_layer))
Pred <- t(Pred)
Pred <- apply(Pred , 1, FUN=which.max)

error_NN = cv_error_NN <- length(which(Y_vector != Pred))/length(Y_vector)
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



###############################################################

# Since our results were extremely bad, we decided to try "helping" the algorythms and see how high we can get
# To help them we decided to reduce the number of classes by creating clusters.
# We used the MAE we got from the regression on the same dataset, which was a bit more than 2,
# and then took one MAE to the left and one to the right to create the clusters.

rm(list=ls())

# we load the covariate matrix with dummy variables that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix_reg.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_regression.RData") # the name of this vector is Y_vector

# we start by approximating to 19 every number of accidents bigger than it

sum(Y_vector > 19)/length(Y_vector) # 0.35%
Y_vector[which(Y_vector > 19)] <- 19

Y_vector[which(Y_vector <= 4)] <- 1
Y_vector[which(Y_vector >= 5 & Y_vector <= 9)] <- 2
Y_vector[which(Y_vector >= 10 & Y_vector <= 14)] <- 3
Y_vector[which(Y_vector >= 15 & Y_vector <= 19)] <- 4

# this time we don't optimize, as the hyperparameters shoudn't change

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

cv_error_log = length(which(Y_vector != y_classified_log))/length(Y_vector)
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

# We now run a 10-fold cv with the best k we found in the optimization

k_best = 54
fold <- 10
cv_error_knn <- rep(NA, fold)
y_classified_knn <- rep(NA, nrow(X_matrix))

for (i in 1:fold){
  
  X_k <- X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold))),]
  Y_k <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold)))]
  X_test <- X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)),]
  Y_test <- Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))]
  
  pred <- as.numeric(knn(X_k, X_test, cl = Y_k, k = k_best))
  
  y_classified_knn[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))] <- pred
  cv_error_knn[i] <- length(which(Y_test != pred))/length(Y_test)
  
}

cv_error_knn <- mean(cv_error_knn)
print(cv_error_knn)

misclassification_matrix_knn = matrix(0, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
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

# 10-folds cross-validation with optimized parameters

best_cases <- 4
best_CF <- .5

fold = 10
y_classified_boosting <- rep(NA, length(Y_vector))
cv_error_boosting <- rep(NA, fold)

for (i in 1:fold) {
  
  y_train = Y_vector[-((1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold))]
  x_train = X_matrix[-((1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)),]
  x_test = data.frame(X_matrix[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold),])
  y_test = Y_vector[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)]
  
  model = maboost(x = x_train, y = y_train, iter = 1, verbose = 1 ,nu = .1, C50tree = T, C5.0Control(CF = best_CF, minCase = best_cases))
  
  pred = predict(model, x_test, type="class")
  
  y_classified_boosting[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)] <- pred
  cv_error_boosting[i] <- length(which(y_test != as.double(pred)))/length(y_test)
  
}

cv_error_boosting <- mean(cv_error_boosting)
print(cv_error_boosting)
misclassification_matrix_boosting = matrix(0, num_degrees, num_degrees)

for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_boosting[i ,j] = length(which((Y_vector == i) & (y_classified_boosting == j))) / length(which((Y_vector == i)))
  }
}

print(misclassification_matrix_boosting)
cv_error_boosting <- mean(error_boosting)

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

Y_categorized <- to_categorical(Y_vector-1)  # keras requires a specific kind of vector as target, at the same time the indexing in python starts from 0

build_model <- function(){
  
  model <- keras_model_sequential() %>%
    layer_dense(units = ncol(X_matrix), activation = "sigmoid",
                input_shape = ncol(X_matrix)) %>%
    layer_dense(units = 128, activation = "sigmoid") %>%
    layer_dense(units = ncol(Y_categorized), activation = "sigmoid")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(),
    metrics = list("accuracy")
  )
  model
  
}

best_epochs = 15

fold <- 10
cv_error_NN <- rep(NA, fold)
y_classified_NN<- rep(NA, nrow(X_matrix))

for (i in 1:fold){
  
  X_train<- X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold))),]
  Y_train <- Y_categorized[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold))),]
  X_test <- X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)),]
  Y_test <- Y_categorized[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))]
  
  model <- build_model()
  
  training <- model %>% fit(
    X_train, Y_train, 
    epochs = best_epochs, batch_size = 128,
    validation.split = 0,
    verbose =  1)
  
  pred <- model %>% predict_classes(X_test)
  
  cv_error_NN[i] <- length(which(Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*nrow(X_matrix)/fold)] != pred + 1))
  y_classified_NN[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))] <- pred+1
  
}

cv_error_NN <- sum(cv_error_NN)/length(Y_vector)
print(cv_error_NN)

misclassification_matrix_NN = matrix(0, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_NN[i ,j] = length(which((Y_vector == i) & (y_classified_NN == j))) / length(which((Y_vector == i)))
  }
}
print(misclassification_matrix_knn)

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
