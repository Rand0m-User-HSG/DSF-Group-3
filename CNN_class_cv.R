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
  epochs = 50, batch_size = 1024, 
  validation_split = 0.2)

pred <- model %>% predict_classes(X_matrix)

misclassification_matrix = matrix(0, ncol(Y)-1, ncol(Y)-1)
for (i in 2:ncol(Y)) {
  for (j in 1:(ncol(Y)-1)) {
    misclassification_matrix[j ,i-1] = length(which((Y[,i] == 1) & (pred == i-1))) / length(which((Y[,i] == 1)))
  }
}

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
