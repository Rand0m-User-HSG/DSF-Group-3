#### Introduction ####
# In this script we create a dataset specifically for the visualisation of regression tasks
# The idea is twofold
# (1) rank the rows by dates, so as to be able to plot time series of the number of accidents
# (2) convert all categorical variables (e.g. road type) into dummy variables (1 or 0)
# In doing so we avoid weigthing problems
# e.g. we avoid that the model gives too much weight to the variable "road type" simply because this variable ranges from 1 to 6


rm(list=ls())
library(tidyverse)
library(lubridate)

# we load the dataset for regression tasks that was produced in Data_wrangling.R
load("./Data/data_reg.RData") # this dataset is loaded as df_reg


#### rank the dataset by dates ####
# we first rank the dataset by dates, from the oldest to the most recent

df_reg = df_reg %>% 
  mutate(date = make_date(AccidentYear, AccidentMonth, days))

df_reg = df_reg[order(df_reg$date),]


#### Y_vector ####
# we now create the Y_vector with the number of accidents in canton ZH on each day

Y_vector = df_reg$number_accidents

save(Y_vector, file = "Data/Y_vector_regression.RData")


#### X_matrix ####
# we now create the X_matrix of covariates

# we first drop the target column (Y)
X_matrix = df_reg %>%
  dplyr::select(-number_accidents)

# we then drop the AccidentYear column, because it would not lend itself to future-oriented predictions
# i.e. it would not be suitable if we were to use this model in the future, on another dataset
X_matrix = X_matrix %>%
  dplyr::select(-AccidentYear)

# we then convert the months into dummy variables (1 or 0)
X_matrix = X_matrix %>%
  mutate(January = ifelse(AccidentMonth == 1, 1, 0),
         February = ifelse(AccidentMonth == 2, 1, 0),
         March = ifelse(AccidentMonth == 3, 1, 0),
         April = ifelse(AccidentMonth == 4, 1, 0),
         May = ifelse(AccidentMonth == 5, 1, 0),
         June = ifelse(AccidentMonth == 6, 1, 0),
         July = ifelse(AccidentMonth == 7, 1, 0),
         August = ifelse(AccidentMonth == 8, 1, 0),
         September = ifelse(AccidentMonth == 9, 1, 0),
         October = ifelse(AccidentMonth == 10, 1, 0),
         November = ifelse(AccidentMonth == 11, 1, 0),
         December = ifelse(AccidentMonth == 12, 1, 0)) %>%
  dplyr::select(11:22, 2:10)

# we then convert the weekdays into dummy variables (1 or 0)
X_matrix = X_matrix %>%
  mutate(Monday = ifelse(week_day_number == 1, 1, 0),
         Tuesday = ifelse(week_day_number == 2, 1, 0),
         Wednesday = ifelse(week_day_number == 3, 1, 0),
         Thursday = ifelse(week_day_number == 4, 1, 0),
         Friday = ifelse(week_day_number == 5, 1, 0),
         Saturday = ifelse(week_day_number == 6, 1, 0),
         Sunday = ifelse(week_day_number == 7, 1, 0)) %>%
  dplyr::select(1:14, 22:28, 16:21)

# we then drop the column days, because it is not relevant for our predictions
# indeed what matter is the month and the weekday, but not the day number in the month
X_matrix = X_matrix %>% 
  dplyr::select(-days)

# we then drop the column CantonCode, because it is redundant in a ZH-only dataset
X_matrix = X_matrix %>% 
  dplyr::select(-CantonCode)

# we then standardize the 5 non-dummy columns (Temp, Pressure, Humidity, Prec_Amount, Wind_Spd)

standFun = function(x){ #standFun is a function that standardizes -> mean = 0, sd = 1
  out = (x - mean(x))/sd(x)
  return(out)
}

for(i in 20:24){
  X_matrix[, i] = standFun(X_matrix[, i])
}

# finally we convert the X_matrix (currently a dataframe) into a real matrix
# and split it into a covariate matrix, and a date vector
dates_regression = X_matrix$date
X_matrix = data.matrix(X_matrix[, -25])

save(X_matrix, file = "Data/covariate_matrix_reg.RData")
save(dates_regression, file = "Data/dates_regression.RData")

