#### Introduction ####

# In this script we perform a linear regression
# The goal is to predict the number of accidents in canton ZH on a particular day
# We test our results by first performing 10-fold cross-validation,
# and then leave-one-out cross-validation
# Finally we plot our results


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


#### visualisation of results ####

# in this section we plot some graphs to visualise our results


# we first identify the row index associated with the lowest MSE_LOO_cv
index = head(order(LOO_cv_MSEs), n = 1L)

# and verify that this is the same row index as the one associated with the lowest MAE_LOO_cv
index == head(order(LOO_cv_MAEs), n = 1L) # TRUE

# we then pick the betas associated with this particular index
betas_index = betas_LOO_cv[, index]

# we now prepare a covariate matrix with an intercept
X_matrix = dataset_reg[, 2:25] %>% # eliminate the column "Y_vector" (target)
  mutate(intercept = 1) %>% # add an intercept column
  select(intercept, everything()) # place the intercept column right at the beginning

# for visualisation purposes we restrict our observations to some periods
# and not to the entire 2011-2018 period (which would yield an over-stacked plot with 2897 points)
# we arbitrarily restrict our visualisation to the following periods
# (1) year 2018
# (2) Q1 2011 (Jan-Mar)
# (3) Q2 2013 (Apr-Jun)
# (4) Q3 2015 (Jul-Sep)
# (5) Q4 2017 (Oct-Dec)
# (6) January 2012
# (7) April 2014
# (8) July 2016
# (9) October 2018
# with this selection we make sure to consider every year in the 2011-2018 period,
# as well as different time intervals (yearly, quarterly, monthly)

# we start by loading the dates associated with each row of the Y_vector
# these dates were computed in the script REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/dates_regression.RData") # data loaded as dates_regression

# we select the rows of dates_regression that correspond to the selected periods
dates_2018 = dates_regression[which(year(dates_regression) == 2018)]
dates_Q1_2011 = dates_regression[which(year(dates_regression) == 2011 & month(dates_regression) %in% c(01, 02, 03))]
dates_Q2_2013 = dates_regression[which(year(dates_regression) == 2013 & month(dates_regression) %in% c(04, 05, 06))]
dates_Q3_2015 = dates_regression[which(year(dates_regression) == 2015 & month(dates_regression) %in% c(07, 08, 09))]
dates_Q4_2017 = dates_regression[which(year(dates_regression) == 2017 & month(dates_regression) %in% c(10, 11, 12))]
dates_Jan_2012 = dates_regression[which(year(dates_regression) == 2012 & month(dates_regression) == 01)]
dates_Apr_2014 = dates_regression[which(year(dates_regression) == 2014 & month(dates_regression) == 04)]
dates_Jul_2016 = dates_regression[which(year(dates_regression) == 2016 & month(dates_regression) == 07)]
dates_Oct_2018 = dates_regression[which(year(dates_regression) == 2018 & month(dates_regression) == 10)]

# and based on these dates we create separate covariate matrices and Y_vectors
X_matrix_2018 = X_matrix[which(year(dates_regression) == 2018), ]
X_matrix_Q1_2011 = X_matrix[which(year(dates_regression) == 2011 & month(dates_regression) %in% c(01, 02, 03)), ]
X_matrix_Q2_2013 = X_matrix[which(year(dates_regression) == 2013 & month(dates_regression) %in% c(04, 05, 06)), ]
X_matrix_Q3_2015 = X_matrix[which(year(dates_regression) == 2015 & month(dates_regression) %in% c(07, 08, 09)), ]
X_matrix_Q4_2017 = X_matrix[which(year(dates_regression) == 2017 & month(dates_regression) %in% c(10, 11, 12)), ]
X_matrix_Jan_2012 = X_matrix[which(year(dates_regression) == 2012 & month(dates_regression) == 01), ]
X_matrix_Apr_2014 = X_matrix[which(year(dates_regression) == 2014 & month(dates_regression) == 04), ]
X_matrix_Jul_2016 = X_matrix[which(year(dates_regression) == 2016 & month(dates_regression) == 07), ]
X_matrix_Oct_2018 = X_matrix[which(year(dates_regression) == 2018 & month(dates_regression) == 10), ]

Y_vector_2018 = Y_vector[which(year(dates_regression) == 2018)]
Y_vector_Q1_2011 = Y_vector[which(year(dates_regression) == 2011 & month(dates_regression) %in% c(01, 02, 03))]
Y_vector_Q2_2013 = Y_vector[which(year(dates_regression) == 2013 & month(dates_regression) %in% c(04, 05, 06))]
Y_vector_Q3_2015 = Y_vector[which(year(dates_regression) == 2015 & month(dates_regression) %in% c(07, 08, 09))]
Y_vector_Q4_2017 = Y_vector[which(year(dates_regression) == 2017 & month(dates_regression) %in% c(10, 11, 12))]
Y_vector_Jan_2012 = Y_vector[which(year(dates_regression) == 2012 & month(dates_regression) == 01)]
Y_vector_Apr_2014 = Y_vector[which(year(dates_regression) == 2014 & month(dates_regression) == 04)]
Y_vector_Jul_2016 = Y_vector[which(year(dates_regression) == 2016 & month(dates_regression) == 07)]
Y_vector_Oct_2018 = Y_vector[which(year(dates_regression) == 2018 & month(dates_regression) == 10)]

# now we compute the predictions for the selected periods
pred_2018 = as.matrix(X_matrix_2018) %*% betas_index
pred_Q1_2011 = as.matrix(X_matrix_Q1_2011) %*% betas_index
pred_Q2_2013 = as.matrix(X_matrix_Q2_2013) %*% betas_index
pred_Q3_2015 = as.matrix(X_matrix_Q3_2015) %*% betas_index
pred_Q4_2017 = as.matrix(X_matrix_Q4_2017) %*% betas_index
pred_Jan_2012 = as.matrix(X_matrix_Jan_2012) %*% betas_index
pred_Apr_2014 = as.matrix(X_matrix_Apr_2014) %*% betas_index
pred_Jul_2016 = as.matrix(X_matrix_Jul_2016) %*% betas_index
pred_Oct_2018 = as.matrix(X_matrix_Oct_2018) %*% betas_index

# finally we plot the dates, the predictions and the actual values (Y_vector)
library(grDevices)
transparent_blue = rgb(0,0,1, alpha = 0.8)

plot(dates_2018, pred_2018, type = "l", col = "red", main = "Accidents in 2018", sub = "daily number of accidents in canton ZH in 2018", xlab = "", ylab = "number of accidents")
points(dates_2018, Y_vector_2018, col = transparent_blue, cex = 0.5, xlab = "")

plot(dates_Q1_2011, pred_Q1_2011, type = "l", col = "red", main = "Accidents in Q1 2011", sub = "daily number of accidents in canton ZH in Q1 2011 (Jan-Mar)", xlab = "", ylab = "number of accidents")
points(dates_Q1_2011, Y_vector_Q1_2011, col = transparent_blue, cex = 0.5, xlab = "")

plot(dates_Q2_2013, pred_Q2_2013, type = "l", col = "red", main = "Accidents in Q2 2013", sub = "daily number of accidents in canton ZH in Q2 2013 (Apr-Jun)", xlab = "", ylab = "number of accidents")
points(dates_Q2_2013, Y_vector_Q2_2013, col = transparent_blue, cex = 0.5, xlab = "")

plot(dates_Q3_2015, pred_Q3_2015, type = "l", col = "red", main = "Accidents in Q3 2015", sub = "daily number of accidents in canton ZH in Q3 2015 (Jul-Sep)", xlab = "", ylab = "number of accidents")
points(dates_Q3_2015, Y_vector_Q3_2015, col = transparent_blue, cex = 0.5, xlab = "")

plot(dates_Q4_2017, pred_Q4_2017, type = "l", col = "red", main = "Accidents in Q4 2017", sub = "daily number of accidents in canton ZH in Q4 2017 (Oct-Dec)", xlab = "", ylab = "number of accidents")
points(dates_Q4_2017, Y_vector_Q4_2017, col = transparent_blue, cex = 0.5, xlab = "")

plot(dates_Jan_2012, pred_Jan_2012, type = "l", col = "red", main = "Accidents in January 2012", sub = "daily number of accidents in canton ZH in January 2012", xlab = "", ylab = "number of accidents")
points(dates_Jan_2012, Y_vector_Jan_2012, col = transparent_blue, cex = 0.5, xlab = "")

plot(dates_Apr_2014, pred_Apr_2014, type = "l", col = "red", main = "Accidents in April 2014", sub = "daily number of accidents in canton ZH in April 2014", xlab = "", ylab = "number of accidents")
points(dates_Apr_2014, Y_vector_Apr_2014, col = transparent_blue, cex = 0.5, xlab = "")

plot(dates_Jul_2016, pred_Jul_2016, type = "l", col = "red", main = "Accidents in July 2016", sub = "daily number of accidents in canton ZH in July 2016", xlab = "", ylab = "number of accidents")
points(dates_Jul_2016, Y_vector_Jul_2016, col = transparent_blue, cex = 0.5, xlab = "")

plot(dates_Oct_2018, pred_Oct_2018, type = "l", col = "red", main = "Accidents in October 2018", sub = "daily number of accidents in canton ZH in October 2018", xlab = "", ylab = "number of accidents")
points(dates_Oct_2018, Y_vector_Oct_2018, col = transparent_blue, cex = 0.5, xlab = "")
