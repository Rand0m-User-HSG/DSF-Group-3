#### Introduction ####

# In this script we perform a linear regression
# The goal is to predict the number of accidents in canton ZH on a particular day
# We test our results by first performing 10-fold cross-validation,
# and then leave-one-out cross-validation
# Finally we re-do the same steps with a more sophisticated linear regression
# that includes interaction effects between weather variables 


rm(list=ls())
library(tidyverse)
library(modelr)
library(broom) 


# we load the covariate matrix with dummy variables that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/covariate_matrix_reg.RData") # the name of this matrix is X_matrix

# we load the Y_vector that was produced in REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_regression.RData") # the name of this vector is Y_vector


#### linear regression ####

# in this section we perform a simple linear regression
# without caring about cross-validation yet

# we first create the dataset needed for linear regression
dataset_reg = as.data.frame(cbind(Y_vector, X_matrix))

# we then perform the linear regression
model_reg = lm(Y_vector ~ ., data = dataset_reg)

# and we calculate the associated MSE and MAE
dataset_reg %>%
  add_predictions(model_reg) %>%
  summarise(MSE = mean((Y_vector - pred)^2),
            MAE = mean(abs(Y_vector - pred)))

# MSE_regression = 9.362053
# MAE_regression = 2.376562


#### 10-fold cross-validation ####

# in this section we perform a linear regression
# and carry out a 10-fold cross-validation

fold = 10
betas_10_fold_cv = matrix(0, nrow = ncol(dataset_reg), ncol = fold)
pred = rep(0, nrow(dataset_reg))
ten_fold_cv_MSEs = rep(0, fold)
ten_fold_cv_MAEs = rep(0, fold)

for (i in 1:fold){
  
  lower_bound_i = (i-1)*(round(nrow(X_matrix)/fold))+1
  upper_bound_i = round(nrow(X_matrix)/fold*i)
  
  dataset_reg_cv_i = dataset_reg[(lower_bound_i:upper_bound_i), ]
  dataset_reg_cv_non_i = dataset_reg[-(lower_bound_i:upper_bound_i), ]
  y_vector_i = dataset_reg_cv_i[, 1]
  
  model_reg_cv = lm(Y_vector ~ ., data = dataset_reg_cv_non_i)
  coefficients_cv = model_reg_cv$coefficients
  for(j in 1:length(coefficients_cv)){
    coefficients_cv[j] = ifelse(is.na(coefficients_cv[j]), 0, coefficients_cv[j])
  }
  betas_10_fold_cv[, i] = coefficients_cv
  
  dataset_reg_cv_i = dataset_reg_cv_i[, 2:ncol(dataset_reg_cv_i)] %>% # eliminate the column "Y_vector" (target)
    mutate(intercept = 1) %>% # add an intercept column
    select(intercept, everything())
  
  pred[(lower_bound_i:upper_bound_i)] = as.matrix(dataset_reg_cv_i) %*% betas_10_fold_cv[, i]
  # we use dataset_reg_cv_non_i to calculate the betas, and test these betas on fold i (dataset_reg_cv_i)
  
  ten_fold_cv_MSEs[i] = mean((y_vector_i - pred[(lower_bound_i:upper_bound_i)])^2)
  ten_fold_cv_MAEs[i] = mean(abs(y_vector_i - pred[(lower_bound_i:upper_bound_i)]))
  
}

MSE_10_fold_cv = mean(ten_fold_cv_MSEs)
MAE_10_fold_cv = mean(ten_fold_cv_MAEs)

# MSE_10_fold_cv = 9.660287 (MSE_regression  = 9.362053)
# MAE_10_fold_cv = 2.416092 (MAE_regression = 2.376562)

# Getting slightly higher cv errors is normal, because they correspond to "testing" errors
# unlike the errors from the previous section which were merely "training" errors


#### Leave-one-out cross-validation ####

# in this section we perform a linear regression
# and carry out a leave-one-out (LOO) cross-validation

fold_LOO = nrow(dataset_reg)
betas_LOO_cv = matrix(0, nrow = ncol(dataset_reg), ncol = fold_LOO)
pred = rep(0, fold_LOO)
LOO_cv_MSEs = rep(0, fold_LOO)
LOO_cv_MAEs = rep(0, fold_LOO)

for (i in 1:fold_LOO){
  
  dataset_reg_cv_i = dataset_reg[i, ]
  dataset_reg_cv_non_i = dataset_reg[-i, ]
  y_vector_i = dataset_reg_cv_i[, 1]
  
  model_reg_cv = lm(Y_vector ~ ., data = dataset_reg_cv_non_i)
  coefficients_cv = model_reg_cv$coefficients
  for(j in 1:length(coefficients_cv)){
    coefficients_cv[j] = ifelse(is.na(coefficients_cv[j]), 0, coefficients_cv[j])
  }
  betas_LOO_cv[, i] = coefficients_cv
  
  dataset_reg_cv_i = dataset_reg_cv_i[, 2:ncol(dataset_reg_cv_i)] %>% # eliminate the column "Y_vector" (target)
    mutate(intercept = 1) %>% # add an intercept column
    select(intercept, everything()) # place the intercept column right at the beginning
  
  pred[i] = as.matrix(dataset_reg_cv_i) %*% betas_LOO_cv[, i]
  # we use dataset_reg_cv_non_i to calculate the betas, and test these betas on fold i (dataset_reg_cv_i)
  
  LOO_cv_MSEs[i] = (y_vector_i - pred[i])^2
  LOO_cv_MAEs[i] = abs(y_vector_i - pred[i])
  
}

MSE_LOO_cv = mean(LOO_cv_MSEs)
MAE_LOO_cv = mean(LOO_cv_MAEs)

# MSE_LOO_cv = 9.514341 (MSE_10_fold_cv = 9.660287; MSE_regression  = 9.362053)
# MAE_LOO_cv = 2.395989 (MAE_10_fold_cv = 2.416092; MAE_regression = 2.376562)

# Getting slightly lower leave-one-out cv errors compared to 10-fold cv is normal, because
# leave-one-out cv provides for a more refined model (we get 2897 iterations, not just 10)


#### linear regression with interaction effects ####

# in this section we perform a more sophisticated linear regression
# that includes interaction effects between weather variables;
# we do not care about cross-validation yet

# we first create the dataset needed for linear regression
dataset_reg = as.data.frame(cbind(Y_vector, X_matrix))

# we then perform the linear regression
model_reg = lm(Y_vector ~ . + Temp*Pressure + Temp*Humidity + Temp*Prec_amount + Temp*Wind_Spd
               + Pressure*Humidity + Pressure*Prec_amount + Pressure*Wind_Spd
               + Humidity*Prec_amount + Humidity*Wind_Spd
               + Prec_amount*Wind_Spd,
               data = dataset_reg)

# and we calculate the associated MSE and MAE
dataset_reg %>%
  add_predictions(model_reg) %>%
  summarise(MSE = mean((Y_vector - pred)^2),
            MAE = mean(abs(Y_vector - pred)))

# MSE_regression_interaction = 9.254047
# MAE_regression_interaction = 2.364247

# Incorporating interaction effects slightly reduces MSE and MAE


#### 10-fold cross-validation with interaction effects ####

# in this section we perform a linear regression with interaction effects
# and carry out a 10-fold cross-validation

fold = 10
pred = rep(0, nrow(dataset_reg))
ten_fold_cv_MSEs_interaction = rep(0, fold)
ten_fold_cv_MAEs_interaction = rep(0, fold)

for (i in 1:fold){
  
  lower_bound_i = (i-1)*(round(nrow(X_matrix)/fold))+1
  upper_bound_i = round(nrow(X_matrix)/fold*i)
  
  dataset_reg_cv_i = dataset_reg[(lower_bound_i:upper_bound_i), ]
  dataset_reg_cv_non_i = dataset_reg[-(lower_bound_i:upper_bound_i), ]
  y_vector_i = dataset_reg_cv_i[, 1]
  
  model_reg_cv = lm(Y_vector ~ . + Temp*Pressure + Temp*Humidity + Temp*Prec_amount + Temp*Wind_Spd
                    + Pressure*Humidity + Pressure*Prec_amount + Pressure*Wind_Spd
                    + Humidity*Prec_amount + Humidity*Wind_Spd
                    + Prec_amount*Wind_Spd,
                    data = dataset_reg_cv_non_i)
  
  dataset_reg_cv_i_with_pred = dataset_reg_cv_i %>%
    add_predictions(model_reg_cv)
  
  pred[(lower_bound_i:upper_bound_i)] = dataset_reg_cv_i_with_pred$pred
  # we use dataset_reg_cv_non_i to calculate the betas, and test these betas on fold i (dataset_reg_cv_i)
  
  ten_fold_cv_MSEs_interaction[i] = mean((y_vector_i - pred[(lower_bound_i:upper_bound_i)])^2)
  ten_fold_cv_MAEs_interaction[i] = mean(abs(y_vector_i - pred[(lower_bound_i:upper_bound_i)]))
  
}

MSE_10_fold_cv_interaction = mean(ten_fold_cv_MSEs_interaction)
MAE_10_fold_cv_interaction = mean(ten_fold_cv_MAEs_interaction)

# MSE_10_fold_cv_interaction = 9.62265 (MSE_10_fold_cv = 9.660287)
# MAE_10_fold_cv_interaction = 2.412156 (MAE_10_fold_cv = 2.416092)

# Incorporating interaction effects slightly reduces MSE and MAE


#### Leave-one-out cross-validation with interaction effects ####

# in this section we perform a linear regression with interaction effects
# and carry out a leave-one-out (LOO) cross-validation

fold_LOO = nrow(dataset_reg)
pred = rep(0, fold_LOO)
LOO_cv_MSEs_interaction = rep(0, fold_LOO)
LOO_cv_MAEs_interaction = rep(0, fold_LOO)

for (i in 1:fold_LOO){
  
  dataset_reg_cv_i = dataset_reg[i, ]
  dataset_reg_cv_non_i = dataset_reg[-i, ]
  y_vector_i = dataset_reg_cv_i[, 1]
  
  model_reg_cv = lm(Y_vector ~ . + Temp*Pressure + Temp*Humidity + Temp*Prec_amount + Temp*Wind_Spd
                    + Pressure*Humidity + Pressure*Prec_amount + Pressure*Wind_Spd
                    + Humidity*Prec_amount + Humidity*Wind_Spd
                    + Prec_amount*Wind_Spd,
                    data = dataset_reg_cv_non_i)
  
  dataset_reg_cv_i_with_pred = dataset_reg_cv_i %>%
    add_predictions(model_reg_cv)
 
  pred[i] = dataset_reg_cv_i_with_pred$pred
  # we use dataset_reg_cv_non_i to calculate the betas, and test these betas on fold i (dataset_reg_cv_i)
  
  LOO_cv_MSEs_interaction[i] = (y_vector_i - pred[i])^2
  LOO_cv_MAEs_interaction[i] = abs(y_vector_i - pred[i])
  
}

MSE_LOO_cv_interaction = mean(LOO_cv_MSEs_interaction)
MAE_LOO_cv_interaction = mean(LOO_cv_MAEs_interaction)

# MSE_LOO_cv_interaction = 9.475153 (MSE_LOO_cv = 9.514341)
# MAE_LOO_cv_interaction =  2.39222 (MAE_LOO_cv = 2.395989)

# Incorporating interaction effects slightly reduces MSE and MAE
