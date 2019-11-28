graphics.off()
rm(list=ls())

# install.packages("smotefamily")
library(smotefamily)

load("./Data/covariate_matrix.RData")
load("./Data/Y_vector_classification.RData")

################################################################### we start by approximating new datapoints


df<- data.frame(cbind(Y_vector, X_matrix))
syn <- SMOTE(df, target = Y_vector, K = 5, dup_size = 19)

syn_data <- as_tibble(syn$data)
syn_data <- syn_data[,1:(ncol(syn_data)-1)]

df_new <- rbind(df, syn_data[which(syn_data$Y_vector != 3),])

nrow(df_new[which(df_new$Y_vector == 3),]) # we now have a dataset with 55% ligth_injuries, 29% severe and 16% fatalities
nrow(df_new[which(df_new$Y_vector == 2),])
nrow(df_new[which(df_new$Y_vector == 1),])


resh <- sample(nrow(df_new))
df_new <- df_new[resh,]
Y_syn <- df_new$Y_vector
X_syn <- df_new[,-1]

#################################################################### we can now try our models again, starting by the knn

library(class)

fold <- 10
cv_error_knn <- rep(NA, fold)
y_classified <- rep(NA, nrow(X_matrix))

for (i in 1:fold){
  
  X_k <- X_syn[(1 + (i-1)*nrow(X_syn)/fold):(i*(nrow(X_syn)/fold)),]
  Y_k <- Y_syn[(1 + (i-1)*nrow(X_syn)/fold):(i*(nrow(X_syn)/fold))]
  X_test <- X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))),]
  Y_test <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)))]
  
  pred <- as.numeric(knn(X_k, X_test, cl = Y_k, k = 9))
  
  y_classified[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)))] <- pred
  cv_error_knn[i] <- mean(abs(Y_test - pred))
  
}

print(mean(cv_error_knn))

misclassification_matrix_knn = matrix(0, unique(Y_vector), unique(Y_vector))
for (i in 1:unique(Y_vector)) {
  for (j in 1:unique(Y_vector)) {
    misclassification_matrix_knn[j ,i] = length(which((Y_vector == j) & (y_classified == i))) / length(which((Y_vector == j)))
  }
}
print(misclassification_matrix_knn)
print(mean(cv_error_knn))

######################################################################### now we use boosting

#install.packages("maboost")
#install.packages("caret")
library(rpart)
library(C50)
library(maboost)
library(lattice)
library(tidyverse)
library(caret)

fold = 10
error_cv_boosting = rep(NA, fold)
y_classified <- rep(NA, length(Y_vector))

for (i in 1:fold) {
  
  X_k <- X_syn[(1 + (i-1)*nrow(X_syn)/fold):(i*(nrow(X_syn)/fold)),]
  Y_k <- Y_syn[(1 + (i-1)*nrow(X_syn)/fold):(i*(nrow(X_syn)/fold))]
  X_test <- X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))),]
  Y_test <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)))]
  
  
  model_maboost = maboost(x = data.frame(X_k), y = Y_k, iter = 5, nu = .1, C50tree = T, C5.0Control(CF = .2, minCase = 128))
  
  pred = predict(model_maboost, data.frame(X_test), type="class")
  
  y_classified[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)] <- pred
  error_cv_boosting[i] = mean(abs(Y_test - as.integer(pred)))
}

misclassification_matrix_boosting = matrix(0, unique(Y_vector), unique(Y_vector))
for (i in 1:unique(Y_vector)) {
  for (j in 1:unique(Y_vector)) {
    misclassification_matrix_boosting[j ,i] = length(which((Y_vector == j) & (y_classified == i))) / length(which((Y_vector == j)))
  }
}
print(misclassification_matrix_boosting)
print(mean(error_cv_boosting))

############################################################## time for logistic regression

num_degrees <- unique(Y_vector)
fold = 10
beta_one_vs_all_cv = matrix(0, ncol(X_matrix)+1, num_degrees)
y_classified_cv = rep(0, length(Y_vector))
cv_error_log = rep(NA, length(Y_vector))

for (i in 1:fold) {
  
  X_k <- X_syn[(1 + (i-1)*nrow(X_syn)/fold):(i*(nrow(X_syn)/fold)),]
  Y_k <- Y_syn[(1 + (i-1)*nrow(X_syn)/fold):(i*(nrow(X_syn)/fold))]
  X_test <- X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))),]
  Y_test <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)))] 
  
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
  
  y_classified[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)] = apply(cbind(rep(1,round(nrow(X_matrix)/fold)), X_test) %*% beta_one_vs_all_cv , 1, FUN=which.max)
  cv_error[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)] = abs(Y_vector[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)] - y_classified[(1+(i-1)*nrow(X_matrix)/fold):(i*nrow(X_matrix)/fold)])
  # we use x_non_i and y_non_i to calculate beta, and test this beta on fold i
}

misclassification_matrix_log = matrix(NA, num_degrees, num_degrees)
for (i in 1:num_degrees) {
  for (j in 1:num_degrees) {
    misclassification_matrix_log[i, j] = length(which((Y_vector == i) & (y_classified == j))) / length(which((Y_vector == i)))
  }
}
print(misclassification_matrix_log)
print(mean(cv_error_log))

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

Y <- to_categorical(Y_vector)

fold = 10

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

cv_error_cnn <- rep(NA, fold)
y_classified <- rep(NA, nrow(X_matrix))

for (i in 1:fold){
  
  X_k <- X_syn[(1 + (i-1)*nrow(X_syn)/fold):(i*(nrow(X_syn)/fold)),]
  Y_k <- to_categorical(Y_syn[(1 + (i-1)*nrow(X_syn)/fold):(i*(nrow(X_syn)/fold))])
  X_test <- X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))),]
  Y_test <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)))]
  
  model <- model_build()
  
  training <- model %>% fit(
    X_k, Y_k, 
    epochs = 50, batch_size = 1024,
    validation.split = 0,
    verbose = 1)
  
  pred <- model %>% predict_classes(X_test)
  y_classified[-((1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)))] <- pred
  cv_error_cnn[i] <- mean(abs(Y_test - pred))
  
}

misclassification_matrix_cnn = matrix(0, ncol(Y)-1, ncol(Y)-1)
for (i in 2:ncol(Y)) {
  for (j in 1:(ncol(Y)-1)) {
    misclassification_matrix_cnn[j ,i-1] = length(which((Y[,i] == 1) & (y_classified == i-1))) / length(which((Y[,i] == 1)))
  }
}
print(misclassification_matrix_cnn)
print(mean(cv_error_cnn))

# as expected, the CNN performs better

####################################################### Let's group the various results togheter

# cv_errors

print(paste("cv_error KNN:", mean(cv_error_knn)))
print(paste("cv_error Boosting:", mean(cv_error_boosting)))
print(paste("cv_error Logistic regression:", mean(cv_error_log)))
print(paste("cv_error CNN:", mean(cv_error_cnn)))

# misclassification matrices

print(paste("misclassification matrix KNN:",misclassification_matrix_knn))
print(paste("misclassification matrix Boosting:",misclassification_matrix_boosting))
print(paste("misclassification matrix logistic regression:", misclassification_matrix_log))
print(paste("misclassification matrix CNN:",misclassification_matrix_cnn))