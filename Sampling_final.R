graphics.off()
rm(list=ls())
load("./Data/covariate_matrix.RData")
load("./Data/Y_vector_classification.RData")

#### Introduction ####

# The point of this script is to randomly sample a more balanced dataset, in order to see if us predicting only 
# ligth injuries is caused by the overaboundance of them.
# Once we get our new dataset we use the same 4 models we used before, that is logistic regression, knn, boosting
# and neural network. As we just want to see how the predictions change we avoided optimizing the hyperarameters.

df <- data.frame(cbind(Y_vector, X_matrix))

ligth <- which(df$Y_vector == "3")
severe <- which(df$Y_vector== "2")
fatalities <- which(df$Y_vector == "1")

# we want the fatalities to be 16% and the severe_injuries 29%

ligth_bal <- sample(ligth, size = 55/16*length(fatalities))
severe_bal <- sample(severe, size = 29/16*length(fatalities))

sampled <- c(ligth_bal, severe_bal, fatalities)
resh <- sample(length(sampled))
sampled <- sampled[resh]
X_sample <- X_matrix[sampled,]
Y_sample <- Y_vector[sampled]

#### we can now try our models again, starting by the logistic regression ####

num_degrees <- length(unique(Y_vector))
fold = 10
beta_one_vs_all_cv = matrix(0, ncol(X_matrix)+1, num_degrees)
y_classified_log = rep(0, length(Y_vector))

for (i in 1:fold) {
  
  X_k <- X_sample[-((1 + (i-1)*nrow(X_sample)/fold):(i*(nrow(X_sample)/fold))),]
  Y_k <- Y_sample[-((1 + (i-1)*nrow(X_sample)/fold):(i*(nrow(X_sample)/fold)))]
  X_test <- X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)),]
  Y_test <- Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))]
  
  for (d in 1:num_degrees) {
    
    degree_selected = which(Y_k == d)
    y_d = Y_k
    y_d[-degree_selected] = 0  
    y_d[degree_selected] = 1
    
    data = data.frame(y_d, X_k)
    model_glmfit_d = glm(y_d ~., data, start = rep(0,ncol(X_matrix)+1) ,family=binomial(link="logit"),
                         control = list(maxit = 100, trace = FALSE) )
    beta_glmfit_d  = model_glmfit_d$coefficients
    beta_glmfit_d[is.na(beta_glmfit_d)] = 0
    
    beta_one_vs_all_cv[, d] = beta_glmfit_d 
  }
  
  y_classified_log[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)] = apply(cbind(rep(1,round(nrow(X_matrix)/fold)), X_test) %*% beta_one_vs_all_cv , 1, FUN=which.max)
}

cv_error_log <- length(which(Y_vector != y_classified_log))/length(Y_vector)
print(cv_error_log)

misclassification_matrix_log = matrix(NA, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_log[i, j] = length(which((Y_vector == i) & (y_classified == j))) / length(which((Y_vector == i)))
  }
}

print(misclassification_matrix_log)


#### now we use boosting ####

#install.packages("maboost")
#install.packages("caret")
library(rpart)
library(C50)
library(maboost)
library(lattice)
library(tidyverse)
library(caret)

fold = 10
cv_error_boosting = rep(NA, fold)
y_classified_boosting <- rep(NA, length(Y_vector))

for (i in 1:fold) {
  
  X_k <- X_sample[-((1 + (i-1)*nrow(X_sample)/fold):(i*(nrow(X_sample)/fold))),]
  Y_k <- Y_sample[-((1 + (i-1)*nrow(X_sample)/fold):(i*(nrow(X_sample)/fold)))]
  X_test <- X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)),]
  Y_test <- Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))]
  
  
  model_maboost = maboost(x = data.frame(X_k), y = Y_k, iter = 5, nu = .1, C50tree = T, C5.0Control(CF = .2, minCase = 128))
  
  pred = predict(model_maboost, data.frame(X_test), type="class")
  
  y_classified_boosting[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)] <- pred
}

cv_error_boosting <- length(which(Y_vector != y_classified_boosting))/length(Y_vector)
print(cv_error_boosting)

misclassification_matrix_boosting = matrix(0, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_boosting[j ,i] = length(which((Y_vector == j) & (y_classified == i))) / length(which((Y_vector == j)))
  }
}
print(misclassification_matrix_boosting)

############################################################## time for logistic regression
library(class)

fold <- 10
y_classified_knn <- rep(NA, nrow(X_matrix))
k_best <- sqrt(nrow(X_matrix))

for (i in 1:fold){
  
  X_k <- X_sample[-((1 + (i-1)*nrow(X_sample)/fold):(i*(nrow(X_sample)/fold))),]
  Y_k <- Y_sample[-((1 + (i-1)*nrow(X_sample)/fold):(i*(nrow(X_sample)/fold)))]
  X_test <- X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)),]
  Y_test <- Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))]
  
  pred <- as.numeric(knn(X_k, X_test, cl = Y_k, k = k_best))
  
  y_classified_knn[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))] <- pred

}

cv_error_knn <- length(which(Y_vector != y_classified_knn))/length(Y_vector)
print(mean(cv_error_knn))

misclassification_matrix_knn = matrix(0, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_knn[j ,i] = length(which((Y_vector == j) & (y_classified == i))) / length(which((Y_vector == j)))
  }
}
print(misclassification_matrix_knn)


######################################################################## last but not least, the CNN

#install.packages("tensor")
#install.packages("keras")
#install.packages("tensorflow")
#install.packages("Rcpp")
library(tensor)
library(tensorflow)
library(keras)
library(tidyverse)
#install_tensorflow()

Y_categorized <- to_categorical(Y_vector-1)
Y_sample_categorized <- to_categorical(Y_sample-1)

fold = 10

model_build <- function(){
  
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = ncol(X_sample), activation = "sigmoid", input_shape = ncol(X_sample)) %>% 
    layer_dense(units = 128, activation = "sigmoid") %>%
    layer_dense(units = ncol(Y_categorized), activation = "softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(),
    metrics = c("accuracy")
  )
  model
  
}

y_classified_NN <- rep(NA, nrow(X_matrix))

for (i in 1:fold){
  
  X_k <- X_sample[-((1 + (i-1)*nrow(X_sample)/fold):(i*(nrow(X_sample)/fold))),]
  Y_k <- Y_sample_categorized[-((1 + (i-1)*nrow(X_sample)/fold):(i*(nrow(X_sample)/fold))),]
  X_test <- X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)),]
  Y_test <- Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))]
  
  model <- model_build()
  
  training <- model %>% fit(
    X_k, Y_k, 
    epochs = 5, batch_size = 128,
    validation.split = 0,
    verbose = 1)
  
  pred <- model %>% predict_classes(X_test)
  y_classified_NN[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))] <- pred
  
}

cv_error_NN <- length(which(Y_vector != y_classified_NN+1))/length(Y_vector)
print(cv_error_NN)

misclassification_matrix_NN = matrix(0, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_NN[i ,j] = length(which((Y_vector == i) & (y_classified_NN == j-1))) / length(which((Y_vector == i)))
  }
}
print(misclassification_matrix_NN)


##### Let's group the various results togheter ####

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