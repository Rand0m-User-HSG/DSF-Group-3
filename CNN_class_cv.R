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

load("./Data/covariate_matrix.RData")
load("./Data/Y_vector_classification.RData")

Y <- to_categorical(Y_vector)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = ncol(X_matrix), activation = "sigmoid", input_shape = ncol(X_matrix)) %>% 
  layer_dense(units = 256, activation = "sigmoid") %>%
  layer_dense(units = 4, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_sgd(),
  metrics = c("accuracy")
)

training <- model %>% fit(
  X_matrix, Y, 
  epochs = 50, batch_size = 128, 
  validation_split = 0.2)

pred <- model %>% predict_classes(X_matrix)

misclassification_matrix = matrix(0, ncol(Y)-1, ncol(Y)-1)
for (i in 2:ncol(Y)) {
  for (j in 1:(ncol(Y)-1)) {
    misclassification_matrix[j ,i-1] = length(which((Y[,i] == 1) & (pred == i-1))) / length(which((Y[,i] == 1)))
  }
}

Beta <- model$get_weights()
Beta_input <- Beta[[1]]
Beta_input2 <- Beta[[2]]
Beta_hidden <- Beta[[3]]
Beta_hidden2 <- Beta[[4]]
Beta_output <- Beta[[5]]
Beta_output2 <- Beta[[6]]
############################################################## 10-fold cv now

k = 10

model_build <- function(){
  
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = ncol(X_k), activation = "sigmoid", input_shape = ncol(X_k)) %>% 
    layer_dense(units = 128, activation = "sigmoid") %>%
    layer_dense(units = 4, activation = "softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(),
    metrics = c("accuracy")
  )
  model
  
}

cv_error <- rep(NA, k)
y_classified <- rep(NA, nrow(X_matrix))

for (i in 1:k){
  
  X_k <- X_matrix[(1 + (i-1)*nrow(X_matrix)/k):(i*(nrow(X_matrix)/k)),]
  Y_k <- to_categorical(Y_vector[(1 + (i-1)*nrow(X_matrix)/k):(i*(nrow(X_matrix)/k))])
  X_test <- X_matrix[-((1 + (i-1)*nrow(X_matrix)/k) : (i*(nrow(X_matrix)/k))),]
  Y_test <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/k) : (i*(nrow(X_matrix)/k)))]
  
  model <- model_build()
  
  training <- model %>% fit(
    X_k, Y_k, 
    epochs = 50, batch_size = 1024,
    validation.split = 0,
    verbose = 0)
  
  pred <- model %>% predict_classes(X_test)
  y_classified[-((1 + (i-1)*nrow(X_matrix)/k) : (i*(nrow(X_matrix)/k)))] <- pred
  cv_error[i] <- mean(abs(Y_test - pred))

}

print(mean(cv_error))

misclassification_matrix = matrix(0, ncol(Y)-1, ncol(Y)-1)
for (i in 2:ncol(Y)) {
  for (j in 1:(ncol(Y)-1)) {
    misclassification_matrix[j ,i-1] = length(which((Y[,i] == 1) & (y_classified == i-1))) / length(which((Y[,i] == 1)))
  }
}
print(misclassification_matrix)


###################################################################################

sigmoid <- function(z) {
  x = 1/(1 + exp(z))
}

SoftMax <- function (z){
  x = exp(z) / sum(exp(z))
}

X <- cbind(rep(1, nrow(X_matrix)), X_matrix)
Beta_input_true <- cbind(Beta_input2, t(Beta_input))

A <- sigmoid(Beta_input_true %*% t(X))

A <- t(A)
A <- cbind(rep(1, nrow(A)), A)
Beta_hidden_true <- cbind(Beta_hidden2, t(Beta_hidden))
dim(Beta_hidden_true)
dim(A)
P <- sigmoid(Beta_hidden_true %*% t(A))

P <- t(P)
P <- cbind(rep(1, nrow(P)), P)
Beta_output_true <- cbind(Beta_output2, t(Beta_output))

Pred <- SoftMax(Beta_output_true %*% t(P))
Pred <- t(Pred)

y_classified <- apply(Pred, 1, FUN = which.max)

Errors = mean(abs(Y_vector - (y_classified-1)))
misclassification_matrix = matrix(0, ncol(Y)-1, ncol(Y)-1)
for (i in 2:ncol(Y)) {
  for (j in 1:(ncol(Y)-1)) {
    misclassification_matrix[j ,i-1] = length(which((Y[,i] == 1) & (y_classified == i))) / length(which((Y[,i] == 1)))
  }
}
print(misclassification_matrix)
