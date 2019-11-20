rm(list = ls())
#install.packages("tensor")
#install.packages("keras")
#install.packages("tensorflow")
#install.packages("Rcpp")
library(tensor)
library(tensorflow)
library(keras)
library(tidyverse)
#install_tensorflow()

load("./Data/covariate_matrix_reg.RData")
load("./Data/Y_vector_regression.RData")

X_matrix[,20:24] <- scale(X_matrix[,20:24])
# Y_vector[which(Y_vector >= 15)] <- 15        # this changes little to none, so we don't have a real problem with "outliers"

build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = ncol(X_matrix), activation = "tanh",
                input_shape = ncol(X_matrix)) %>%
    layer_dense(units = 128, activation = "tanh") %>%
    layer_dense(units = 1, activation = "relu")
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_sgd(),
    metrics = list("mean_absolute_error")
  )
  
  model
}

model <- build_model()
model %>% summary()
training <- model %>% fit(
  X_matrix, Y_vector, 
  epochs = 50, batch_size = 1024, 
  validation_split = 0.2)

############################################################## 10-fold cv now

k = 10

model_build <- function(){
  
  model <- keras_model_sequential() %>%
    layer_dense(units = ncol(X_k), activation = "tanh",
                input_shape = ncol(X_k)) %>%
    layer_dense(units = 128, activation = "tanh") %>%
    layer_dense(units = 1, activation = "relu")
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_sgd(),
    metrics = list("mean_absolute_error")
  )
  model
  
}

cv_MAE <- rep(NA, k)
cv_error <- rep(NA, k)

for (i in 1:k){
  
  X_k <- X_matrix[(1 + (i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k),]
  Y_k <- Y_vector [(1 + (i-1)*nrow(X_matrix)/k):(i*nrow(X_matrix)/k)]
  X_test <- X_matrix[-((1 + (i-1)*nrow(X_matrix)/k) : i*nrow(X_matrix)/k),]
  Y_test <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/k) : i*nrow(X_matrix)/k)]
  
  model <- model_build()
  
  training <- model %>% fit(
    X_k, Y_k, 
    epochs = 20, batch_size = 1024,
    validation.split = 0,
    verbose = 0 )
  
  pred <- model %>% predict(X_test)
  cv_MAE[i] <- mean(abs(Y_test - pred))
  
  pred <- round(pred, 0)
  cv_error[i] <- mean(abs(Y_test - pred))
  
}

mean(cv_MAE)
mean(cv_error)