rm(list= ls())

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
library(tensor)
library(tensorflow)
library(keras)
library(tidyverse)
#install_tensorflow()

load("./Data/covariate_matrix.RData")
load("./Data/Y_vector_classification.RData")

# we first create a function that builds our CNN

Y_vector <- to_categorical(Y_vector)  # keras requires a specific kind of vector as target

build_model <- function(){
  
  model <- keras_model_sequential() %>%
    layer_dense(units = ncol(X_matrix), activation = "sigmoid",
                input_shape = ncol(X_matrix)) %>%
    layer_dense(units = 128, activation = "sigmoid") %>%
    layer_dense(units = 4, activation = "softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(),
    metrics = list("accuracy")
  )
  model
  
}

############################################################## 10-fold cv now

optim_CNN <- function(e, X, Y){
  
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
  errors[i] <- optim_CNN(i, X_matrix, Y_vector)
}

best_epochs <- which(errors == min(errors))
print(best_epochs)
print(errors) # this is our best MAE in a 10-fold CV, the epochs were 4 and the MAE 17

######################################################### Let's use the best number of epochs to build a non-cv model

model <- build_model()

training <- model %>% fit(
  X_matrix, Y_vector, 
  epochs = best_epochs, batch_size = 128,
  validation.split = 0,
  verbose =  0)

pred <- model %>% predict(X_matrix)
error <- mean(abs(Y_vector - pred))
print(error)

misclassification_matrix = matrix(0, ncol(Y_vector)-1, ncol(Y_vector)-1)
for (i in 2:ncol(Y_vector)) {
  for (j in 1:(ncol(Y_vector)-1)) {
    misclassification_matrix[j ,i-1] = length(which((Y_vector[,i] == 1) & (pred == j))) / length(which((Y_vector[,i] == 1)))
  }
}
print(misclassification_matrix)

install.packages("rlist")
library(rlist)
weights_reg <- model$get_weights()
list.save(weights_reg, "Data/weights_reg.RData")

##############################################################  here we use the weights of the model to predict with matrix multiplication

rm(list = ls())

load("./Data/covariate_matrix_reg.RData")
load("./Data/Y_vector_regression.RData")
load("./Data/weights_reg.RData")
Beta <- x

Beta_input <- Beta[[1]]
Beta_input2 <- Beta[[2]]
Beta_hidden <- Beta[[3]]
Beta_hidden2 <- Beta[[4]]
Beta_output <- Beta[[5]]
Beta_output2 <- Beta[[6]]

relu = function(z) {
  x = log(1 + exp(z))
}

X <- cbind(rep(1, nrow(X_matrix)), X_matrix)
Beta_input_true <- cbind(Beta_input2, t(Beta_input))

Input_layer <- tanh(Beta_input_true %*% t(X))

Input_layer <- t(Input_layer)
Input_layer<- cbind(rep(1, nrow(Input_layer)), Input_layer)
Beta_hidden_true <- cbind(Beta_hidden2, t(Beta_hidden))

Hidden_layer <- tanh(Beta_hidden_true %*% t(Input_layer))

Hidden_layer <- t(Hidden_layer)
Hidden_layer <- cbind(rep(1, nrow(Hidden_layer)), Hidden_layer)
Beta_output_true <- cbind(Beta_output2, t(Beta_output))

Pred <- relu(Beta_output_true %*% t(Hidden_layer))
Pred <- t(Pred)

MAE = mean(abs(Y_vector - round(Pred, 0)))
print(MAE)
