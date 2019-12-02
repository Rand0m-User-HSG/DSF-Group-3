#### Introduction ####

# In this script we use 4 regression models to predict the number of accidents on a particular
# day in canton ZH.
# (1) linear regression
# (2) LASSO
# (3) boosting
# (4) neural network (NN)
# We test our results by means of cross-validation, and compute the associated MSE and MAE.
# We perform leave-one-out (LOO) cross-validation for linear regression and LASSO, and
# 10-fold cross-validation for neural networks (for reasons of computational time). For
# boosting, we first run 10-fold cross-validation, and optimize our results to find the
# ideal hyperparameters. We then use these hyperparameters to run a leave-one-out
# cross-validation.

rm(list=ls())
library(tidyverse)
library(modelr)
library(broom)
library(glmnet)  

# we load the covariate matrix with dummy variables that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix_reg.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_regression.RData") # the name of this vector is Y_vector


#### LINEAR REGRESSION - Introduction ####

# We begin with a simple linear regression without any interaction effects
# Then we do the same steps with a more sophisticated linear regression, that includes
# interaction effects between weather variables 


#### LINEAR REGRESSION - No interaction effects - Training ####

# We begin with a simple linear regression, without any interaction effects.
# For comparison purposes, we first perform a "training" linear regression, i.e.
# without cross-validation.

# we first create the dataset needed for linear regression
dataset_reg = as.data.frame(cbind(Y_vector, X_matrix))

# we then perform the linear regression
model_lin_reg = lm(Y_vector ~ ., data = dataset_reg)

# and we calculate the associated MSE and MAE
errors_lin_reg = dataset_reg %>%
  add_predictions(model_lin_reg) %>%
  summarise(MSE_lin_reg = mean((Y_vector - pred)^2),
            MAE_lin_reg = mean(abs(Y_vector - pred)))
print(errors_lin_reg)

# MSE_lin_reg = 9.362024
# MAE_lin_reg = 2.37656



#### LINEAR REGRESSION - No interaction effects - LOO cv ####

# We keep working on a simple linear regression, without any interaction effects.
# We now compute testing errors by carrying out a leave-one-out (LOO) cross-validation

fold_LOO = nrow(dataset_reg)
betas_LOO_cv_lin_reg = matrix(0, nrow = ncol(dataset_reg), ncol = fold_LOO)
pred = rep(0, fold_LOO)
MSEs_lin_reg_LOO_cv = rep(0, fold_LOO)
MAEs_lin_reg_LOO_cv = rep(0, fold_LOO)

for (i in 1:fold_LOO){
  
  dataset_reg_cv_i = dataset_reg[i, ]
  dataset_reg_cv_non_i = dataset_reg[-i, ]
  y_vector_i = dataset_reg_cv_i[, 1]
  
  model_lin_reg_LOO_cv = lm(Y_vector ~ ., data = dataset_reg_cv_non_i)
  coefficients_cv = model_lin_reg_LOO_cv$coefficients
  for(j in 1:length(coefficients_cv)){
    coefficients_cv[j] = ifelse(is.na(coefficients_cv[j]), 0, coefficients_cv[j])
  }
  betas_LOO_cv_lin_reg[, i] = coefficients_cv
  
  dataset_reg_cv_i = dataset_reg_cv_i[, 2:ncol(dataset_reg_cv_i)] %>% # eliminate the column "Y_vector" (target)
    mutate(intercept = 1) %>% # add an intercept column
    select(intercept, everything()) # place the intercept column right at the beginning
  
  pred[i] = as.matrix(dataset_reg_cv_i) %*% betas_LOO_cv_lin_reg[, i]
  # we use dataset_reg_cv_non_i to calculate the betas, and test these betas on fold i (dataset_reg_cv_i)
  
  MSEs_lin_reg_LOO_cv[i] = (y_vector_i - pred[i])^2
  MAEs_lin_reg_LOO_cv[i] = abs(y_vector_i - pred[i])
  
}

MSE_lin_reg_LOO_cv = mean(MSEs_lin_reg_LOO_cv)
MAE_lin_reg_LOO_cv = mean(MAEs_lin_reg_LOO_cv)

print(c(MSE_lin_reg_LOO_cv, MAE_lin_reg_LOO_cv))

# MSE_lin_reg_LOO_cv = 9.514310 (MSE_lin_reg = 9.362024)
# MAE_lin_reg_LOO_cv = 2.395987 (MAE_lin_reg = 2.37656)

# Getting slightly higher MSE and MAE cross-validation errors is normal, because they
# correspond to "testing" errors, unlike the errors from the previous section which were
# merely "training" errors


#### LINEAR REGRESSION - Interaction effects - Training ####

# We continue with a more sophisticated linear regression, with interaction effects
# between the weather variables.
# For comparison purposes, we first perform a "training" linear regression, i.e.
# without cross-validation.

# we first perform the linear regression
model_lin_reg_interaction = lm(Y_vector ~ . + Temp*Pressure + Temp*Humidity + Temp*Prec_amount + Temp*Wind_Spd
                               + Pressure*Humidity + Pressure*Prec_amount + Pressure*Wind_Spd
                               + Humidity*Prec_amount + Humidity*Wind_Spd
                               + Prec_amount*Wind_Spd,
                               data = dataset_reg)

# and we calculate the associated MSE and MAE
errors_lin_reg_interaction = dataset_reg %>%
  add_predictions(model_lin_reg_interaction) %>%
  summarise(MSE_lin_reg_interaction = mean((Y_vector - pred)^2),
            MAE_lin_reg_interaction = mean(abs(Y_vector - pred)))
print(errors_lin_reg_interaction)

# MSE_lin_reg_interaction = 9.254078 (MSE_lin_reg = 9.362024)
# MAE_lin_reg_interaction = 2.364248 (MAE_lin_reg = 2.37656)

# Incorporating interaction effects slightly reduces MSE and MAE.


#### LINEAR REGRESSION - Interaction effects - LOO cv ####

# We keep working on a more sophisticated linear regression, with interaction effects
# between weather variables.
# We now compute testing errors by carrying out a leave-one-out (LOO) cross-validation

fold_LOO = nrow(dataset_reg)
pred = rep(0, fold_LOO)
MSEs_lin_reg_LOO_cv_interaction = rep(0, fold_LOO)
MAEs_lin_reg_LOO_cv_interaction = rep(0, fold_LOO)

for (i in 1:fold_LOO){
  
  dataset_reg_cv_i = dataset_reg[i, ]
  dataset_reg_cv_non_i = dataset_reg[-i, ]
  y_vector_i = dataset_reg_cv_i[, 1]
  
  model_lin_reg_interaction_LOO_cv = lm(Y_vector ~ . + Temp*Pressure + Temp*Humidity + Temp*Prec_amount + Temp*Wind_Spd
                                        + Pressure*Humidity + Pressure*Prec_amount + Pressure*Wind_Spd
                                        + Humidity*Prec_amount + Humidity*Wind_Spd
                                        + Prec_amount*Wind_Spd,
                                        data = dataset_reg_cv_non_i)
  
  dataset_reg_cv_i_with_pred = dataset_reg_cv_i %>%
    add_predictions(model_lin_reg_interaction_LOO_cv)
  
  pred[i] = dataset_reg_cv_i_with_pred$pred
  # we use dataset_reg_cv_non_i to calculate the betas, and test these betas on fold i (dataset_reg_cv_i)
  
  MSEs_lin_reg_LOO_cv_interaction[i] = (y_vector_i - pred[i])^2
  MAEs_lin_reg_LOO_cv_interaction[i] = abs(y_vector_i - pred[i])
  
}

MSE_lin_reg_LOO_cv_interaction = mean(MSEs_lin_reg_LOO_cv_interaction)
MAE_lin_reg_LOO_cv_interaction = mean(MAEs_lin_reg_LOO_cv_interaction)

print(c(MSE_lin_reg_LOO_cv_interaction, MAE_lin_reg_LOO_cv_interaction))

# MSE_lin_reg_LOO_cv_interaction = 9.475186 (MSE_lin_reg_LOO_cv = 9.514310)
# MAE_lin_reg_LOO_cv_interaction =  2.392221 (MAE_lin_reg_LOO_cv = 2.395987)

# Incorporating interaction effects slightly reduces MSE and MAE.


#### LASSO - Introduction ####

# We continue with LASSO
# Our goal is to find the optimal lambda,
# and the associated variables that survived variable selection.
# We will then use these variables in a linear regression.

#### LASSO - finding lambda and the associated variables ####

# The following code runs a LASSO regression on our data, with leave-one-out (LOO) cross-
# validation. The glmnet library is used (loaded at the beginning of the file)
model_LASSO_LOO_cv <- cv.glmnet(x=X_matrix, y=Y_vector, nfolds = nrow(X_matrix), family = "gaussian", alpha = 1)

# Find the best lambda, using the "one-standard-deviation" rule
model_LASSO_LOO_cv$lambda.1se

# model_LASSO_LOO_cv$lambda.1se = 0.1423201

# Find the associated MSE
model_LASSO_LOO_cv$cvm[model_LASSO_LOO_cv$lambda == model_LASSO_LOO_cv$lambda.1se] 
# MSE = 9.824736

pred_lasso <- predict(model_LASSO_LOO_cv, X_matrix)
MAE_lasso_cv <- mean(abs(Y_vector - pred_lasso))

print(MAE_lasso_cv)
# 2.42852
# Find the associated variables that survived variable selection
variables_LASSO = coef(model_LASSO_LOO_cv, s="lambda.1se") %>% 
  broom:::tidy.dgCMatrix()
variables_LASSO = variables_LASSO[1]
print(variables_LASSO)


#### LASSO - linear regression with the selected variables ####

# we run a linear regression using only the variables that survived the LASSO selection
# we test our results by performing a leave-one-out (LOO) cross-validation 

dataset_reg <- data.frame(cbind(Y_vector, X_matrix))
fold_LOO = nrow(dataset_reg)
betas_LOO_cv_LASSO_lin_reg = matrix(0, nrow = nrow(variables_LASSO), ncol = fold_LOO)
pred = rep(0, fold_LOO)
MSEs_LASSO_lin_reg_LOO_cv = rep(0, fold_LOO)
MAEs_LASSO_lin_reg_LOO_cv = rep(0, fold_LOO)

for (i in 1:fold_LOO){
  
  dataset_reg_cv_i = dataset_reg[i, ] %>% 
    select(Y_vector, January, February, March, April, June,
           August, September, December, Wednesday, Saturday,
           Sunday, Humidity, Wind_Spd)
  dataset_reg_cv_non_i = dataset_reg[-i, ] %>% 
    select(Y_vector, January, February, March, April, June,
           August, September, December, Wednesday, Saturday,
           Sunday, Humidity, Wind_Spd)
  y_vector_i = dataset_reg_cv_i[, 1]
  
  model_LASSO_lin_reg_LOO_cv = lm(Y_vector ~ ., data = dataset_reg_cv_non_i)
  coefficients_cv = model_LASSO_lin_reg_LOO_cv$coefficients
  for(j in 1:length(coefficients_cv)){
    coefficients_cv[j] = ifelse(is.na(coefficients_cv[j]), 0, coefficients_cv[j])
  }
  betas_LOO_cv_LASSO_lin_reg[, i] = coefficients_cv
  
  dataset_reg_cv_i = dataset_reg_cv_i[, 2:ncol(dataset_reg_cv_i)] %>% # eliminate the column "Y_vector" (target)
    mutate(intercept = 1) %>% # add an intercept column
    select(intercept, everything()) # place the intercept column right at the beginning
  
  pred[i] = as.matrix(dataset_reg_cv_i) %*% betas_LOO_cv_LASSO_lin_reg[, i]
  # we use dataset_reg_cv_non_i to calculate the betas, and test these betas on fold i (dataset_reg_cv_i)
  
  MSEs_LASSO_lin_reg_LOO_cv[i] = (y_vector_i - pred[i])^2
  MAEs_LASSO_lin_reg_LOO_cv[i] = abs(y_vector_i - pred[i])
  
}

MSE_LASSO_lin_reg_LOO_cv = mean(MSEs_LASSO_lin_reg_LOO_cv)
MAE_LASSO_lin_reg_LOO_cv = mean(MAEs_LASSO_lin_reg_LOO_cv)

print(c(MSE_LASSO_lin_reg_LOO_cv, MAE_LASSO_lin_reg_LOO_cv))

# MSE_LASSO_lin_reg_LOO_cv = 9.590648 (MSE_lin_reg_LOO_cv = 9.514310)
# MAE_LASSO_lin_reg_LOO_cv = 2.404960 (MAE_lin_reg_LOO_cv = 2.395987)



#### BOOSTING - Introduction ####

# We now perform a generalized a boosted regresion model 

# install.packages("gbm")
# install.packages("caret")

library(gbm)

optim_boosting<- function(depth, trees, X, Y){
  
  fold = 10
  X <- data.frame(X)
  Y <- as.double((Y))
  
  model_gbm = gbm(Y ~., data = X, distribution = "gaussian", shrinkage = .1, n.trees = trees, 
                  interaction.depth = depth, cv.folds = fold)
  
  pred <- model_gbm$fit
  
  return(mean(abs(Y_vector - pred)))
}

# To optimize this function there're a lot of values we could try, as the number of trees (= n.trees) cna be really big > 10k
# We found out that the optimal number is usually between 3 000 and 10 000 trees, but it's still quite a big range.
# To shrink down the possibilities we start measuring the errors for every thousand trees and tehn we take a range
# around the best number.

# 3000 --> 2.171638
# 4000 --> 2.140644
# 5000 --> 2.119288
# 6000 --> 2.093422
# 7000 --> 2.07393
# 8000 --> 2.031408
# 9000 --> 2.030103
# 10000 --> 2.032644

# It seems that around 9 000 we're gonna find the smallest error, so now we'll search for a range or +/- 100 around it,
# while also modifying the intercation.depth

possible_depth <- 1:5
possible_trees <- 8900:9100

col = 0
error <- matrix(NA, nrow = length(possible_trees), ncol = length(possible_depth))

for (k in possible_depth){
  col = col +1
  row = 0
  for (j in possible_trees){
    row = row +1
    error[row, col] <- optim_boosting(k, j, X_matrix, Y_vector)
  }
}

best_parameters <- which(error == min(error), arr.ind = T)
print(best_parameters)
best_trees <- best_parameters[1]+8900 -1     # we have 8996 trees and
best_depth <- best_parameters[2]             # a intercation depth of 5
print(min(error))

# We'll now run a 10-fol cv for boosting with the optimized parameters, so if you didn't optimize please run the following commented code:
 best_trees = 8996
 best_depth = 5

# we now run a 10-fold cv with the best parameters

fold = 10
MAE_boosting = rep(NA, fold)
MSE_boosting = rep(NA, fold)
y_pred_boosting <- rep(NA, length(Y_vector))

for (i in 1:fold){
  
  X_k <- data.frame(X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold) : i*nrow(X_matrix)/fold),])
  Y_k <- as.double(Y_vector[-((1 + (i-1)*nrow(X_matrix)/fold) : i*nrow(X_matrix)/fold)])
  X_test <- data.frame(X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*nrow(X_matrix)/fold),])
  Y_test <- as.double(Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*nrow(X_matrix)/fold)])
  
  model <- gbm(Y_k ~., data = X_k, distribution = "gaussian", shrinkage = .1, n.trees = best_trees, 
               interaction.depth = best_depth)
  
  pred <- predict(model, X_test, n.trees = best_trees)
  
  y_pred_boosting[(1 + (i-1)*nrow(X_matrix)/fold) : (i*nrow(X_matrix)/fold)] <- pred
  MAE_boosting[i] <- mean(abs(Y_test - pred))
  MSE_boosting[i] = mean((Y_test - pred)^2)
  
}

MSE_boosting <- mean(MSE_boosting)
MAE_boosting <- mean(MAE_boosting)
print(MSE_boosting)
# 0.1435487
print(MAE_boosting)
# 0.2906286
MAE <- mean(abs(Y_vector - y_pred_boosting), na.rm = T)
print(MAE)
# 0.2906286
sum(is.na(y_pred_boosting))
# 7

# No optimization: n.trees = 8996, interaction.depth = 5: 
# MSE_boosting = 0.1435487
# MAE_boosting = 0.2906286

#### NEURAL NETWORK - Introduction ####

# DISCLAIMER: In order to train these models, you'll need to be able to use keras 
# in a python IDE (this means you also need TensorFlow), that is, 
# you also need to (pip) install these packages in python!
# Here is a link to a page in which they explain how to do it: 
# https://towardsdatascience.com/setup-an-environment-for-machine-learning-and-deep-learning-with-anaconda-in-windows-5d7134a3db10
# However, nearing the end of these script, we trained a NN and saved the weights, so that
# even if you aren't able to run the models to get the predictions and the errors, we can get them with 
# highschool-level linear algebra!

#install.packages("tensor")
#install.packages("keras")
#install.packages("tensorflow")
#install.packages("Rcpp")
#install.packages("rlist")
#install.packages("sigmoid")
library(sigmoid)
library(rlist)
library(tensor)
library(tensorflow)
library(keras)
library(tidyverse)
#install_tensorflow()


#### NEURAL NETWORK - Construction of the model ####

# we first create a function that builds our NN

build_model <- function(){
  
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


#### NEURAL NETWORK - optimization with 10-fold cross-validation ####

optim_NN <- function(e, X, Y){
  
  fold = 10
  cv_MAE <- rep(NA, fold)
  
  for (i in 1:fold){
    
    X_k <- X[-((1 + (i-1)*nrow(X)/fold) : i*nrow(X)/fold),]
    Y_k <- Y[-((1 + (i-1)*nrow(X)/fold) : i*nrow(X)/fold)]
    X_test <- X[(1 + (i-1)*nrow(X)/fold) : (i*nrow(X)/fold),]
    Y_test <- Y[(1 + (i-1)*nrow(X)/fold) : (i*nrow(X)/fold)]
    
    model <- build_model()
    
    training <- model %>% fit(
      X_k, Y_k, 
      epochs = e, batch_size = 128,
      validation.split = 0,
      verbose =  0)
    
    pred <- model %>% predict(X_test)
    
    cv_MAE[i] <- mean(abs(Y_test - pred))
    
  }
  
  return(mean(cv_MAE))
}

# since we're training the model with a validation.split of 0, we risk overfitting it, so we search for the epochs,
# which get the smallest MAE on unseen data

possible_epochs = 1:50
errors <- rep(NA, length(possible_epochs))
for (i in possible_epochs){
  errors[i] <- optim_NN(i, X_matrix, Y_vector)
}

best_epochs <- which(errors == min(errors))
print(best_epochs)
print(errors) # this is our best MAE in a 10-fold CV, the epochs were 48 and the MAE 2.444938
# If you haven't run the optimization, please run the following commented code:

# best_epochs = 48

fold <- 10
MAE_NN<- rep(NA, fold)
MSE_NN <- rep(NA, fold)

for (i in 1:fold){
  
  X_train<- X_matrix[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold))),]
  Y_train <- Y_vector[-((1 + (i-1)*nrow(X_matrix)/fold):(i*(nrow(X_matrix)/fold)))]
  X_test <- X_matrix[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold)),]
  Y_test <- Y_vector[(1 + (i-1)*nrow(X_matrix)/fold) : (i*(nrow(X_matrix)/fold))]
  
  model <- build_model()
  
  training <- model %>% fit(
    X_train, Y_train, 
    epochs = best_epochs, batch_size = 128,
    validation.split = 0,
    verbose =  1)
  
  pred <- model %>% predict(X_test)
  
  MSE_NN[i] <- mean((Y_test - pred)^2)
  MAE_NN[i] <- mean(abs(Y_test - pred))
  
}

MSE_NN <- mean(MSE_NN)
MAE_NN <- mean(MAE_NN)
print(MSE_NN)    # 9.71945
print(MAE_NN)    # 2.444938

#### NEURAL NETWORK - Model with the best epochs ####

# Let's use the best number of epochs to build a non-cv model

model <- build_model()

training <- model %>% fit(
  X_matrix, Y_vector, 
  epochs = best_epochs, batch_size = 128,
  validation.split = 0,
  verbose =  0)

pred <- model %>% predict(X_matrix)
MAE <- mean(abs(Y_vector - pred))
print(MAE)  # MAE of 2.302467 

weights_reg <- model$get_weights()
list.save(weights_reg, "Data/weights_reg.RData")


#### NEURAL NETWORK - Predictions with matrix multiplication ####

# here we use the weights of the model to predict with matrix multiplication

load("./Data/weights_reg.RData")
Beta <- x

Beta_input <- Beta[[1]]
Beta_input2 <- Beta[[2]]
Beta_hidden <- Beta[[3]]
Beta_hidden2 <- Beta[[4]]
Beta_output <- Beta[[5]]
Beta_output2 <- Beta[[6]]

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

Pred <- sigmoid::relu(Beta_output_true %*% t(Hidden_layer))
Pred <- t(Pred)

MAE = mean(abs(Y_vector - Pred))
print(MAE)    # the error of 2.302467 is the same as before
