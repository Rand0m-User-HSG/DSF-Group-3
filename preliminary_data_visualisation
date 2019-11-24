#### Introduction ####

# In this script we perform preliminary data visualisation
# We divide these visualisation in three main parts
# (1) geographical visualisation: representing the accident locations and weather stations
# on the map of Switzerland
# (2) number of accidents: reprensenting the distribution of the number of accidents per day
# in canton ZH over the 12 months and the 7 weekdays, and studying its correlation with weather data (precipitation)
# (3) severity of accidents: reprensenting the distribution of the severity of accidents (light injuries, severe injuries, fatalities),
# as well as the share (percentage) of fatal accidents over the 12 months and the 7 weekdays,
# and also studying the correlation of this share of fatal accidents with weather data (precipitation)


rm(list = ls())
library(tidyverse)


#### Geographical visualisation ####

# In this section we produce two maps of Switzerland
# (1) accident locations
# (2) weather stations locations

# we first load the dataset "accidents_location_visualisation.RData"
# that was produced in the script Data_wrangling.R
load("./Data/accidents_location_visualisation.RData") # dataset is loaded under the name "merged" by R

# we then install the required packages for maps
# install.packages("maps")
# install.packages("mapdata")
library(maps)
library(mapdata)
library(grDevices)


# (1) we create a first map that contains all accident locations in Switzerland
# over the period 2011-2018
transparent_red = rgb(1,0,0, alpha = 0.8)
map('worldHires', 'Switzerland')
points(merged$Longitude, merged$Latitude, col = transparent_red, cex = 0.01)


# (2) we then focus on creating a second map that contains all weather stations in Switzerland
# we first create a column "stations_coordinates" that contains both latitude and longitude information
merged = merged %>%
  unite(stations_coordinates, LAT.x, LONG.x)

# then we filter the column to only keep unique values
stations_coordinates = unique(merged$stations_coordinates)

# and separate it back into latitude and longitude values
stations_coordinates = as.data.frame(stations_coordinates) %>%
  separate(stations_coordinates, into = c("latitude", "longitude"), sep = "_")

# we can then create a second map that contains all weather stations in Switzerland
map('worldHires', 'Switzerland')
points(stations_coordinates$longitude, stations_coordinates$latitude, col = "blue", cex = 0.5)


#### number of accidents ####

# In this section we focus on the visualisation of the number of accidents per day in canton ZH
# (1) distribution of the number of accidents per day
# (2) average number of accidents per month
# (3) average number of accidents per weekday
# (4) correlation between precipitation amount and number of accidents


# we start by loading the data that was created for this purpose
# this data consists of "Y_vector_regression.RData" and "covariate_matrix_reg.RData"
# both of them having been created in the script REGRESSION_covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_regression.RData") # data loaded under the name "Y_vector"
load("./Data/covariate_matrix_reg.RData") # data loaded under the name "X_matrix"
dataset = as.data.frame(cbind(Y_vector, X_matrix))


# (1) we visualise the distribution of the number of accidents per day.
# we observe that the number of accidents per day ranges from 0 to 26
min(Y_vector) # min = 0
max(Y_vector) # max = 26

# we thus create a 27-row column vector for this range
distribution_accidents = rep(0, 27)

for(i in 1:27){
  for(j in 1:nrow(dataset)){
    distribution_accidents[i] = ifelse(dataset[j, 1] == i-1, distribution_accidents[i] + 1, distribution_accidents[i])
  }
}

# we verify that the sum of "distribution_accidents" is equal to the number of rows of "dataset"
sum(distribution_accidents) == nrow(dataset) # TRUE

# and now we compute the average distribution of accidents on a yearly basis
# by dividing each row of distribution_accidents by 8
# (because there are 8 years in the period 2011-2018)
average_distribution_accidents = distribution_accidents / 8

# we can now plot the average distribution of accidents
barplot(average_distribution_accidents, names.arg = seq(0, 26),
        main = "Annual distribution of the daily number of accidents", sub = "Average annual distribution of the daily number of accidents in canton ZH (2011-2018)",
        ylim = c(0,40))


# (2) we now visualise the distribution of accidents over 12 months
# i.e. we group the accidents by month
# and compute the mean of each month over the period 2011-2018 (8 years)

total_accidents_per_month = rep(0, 12)

for(i in 1:12){
  for(j in 1:nrow(dataset)){
    total_accidents_per_month[i] = ifelse(dataset[j, i+1] == 1, total_accidents_per_month[i] + dataset[j, 1], total_accidents_per_month[i])
  }
}

# we first verify that the sum of total_accidents_per_month is equal to the sum of the Y_vector column of the dataset
sum(total_accidents_per_month) == sum(dataset$Y_vector) # TRUE

# and now we compute the average number of accidents per month
# by dividing each row of total_accidents_per_month by 8
# (because there are 8 years in the period 2011-2018)
average_accidents_per_month = total_accidents_per_month / 8

# we can now plot the distribution of accidents over 12 months
barplot(average_accidents_per_month, names.arg = str_sub(names(dataset[, 2:13]), 1, 3),
        main = "Average number of accidents per month", sub = "Average number of accidents per month in canton ZH over the period 2011-2018",
        xlim = c(0,12), ylim = c(0,300))


# (3) we now proceed similarly to visualise the distribution of accidents over 7 weekdays
# i.e. we group the accidents by weekday
# and compute the mean of each weekday over the period 2011-2018 (8 years)
total_accidents_per_weekday = matrix(0, nrow = 7, ncol = 2) # add a second column for a counter

for(i in 1:7){
  for(j in 1:nrow(dataset)){
    total_accidents_per_weekday[i, 1] = ifelse(dataset[j, i+13] == 1, total_accidents_per_weekday[i, 1] + dataset[j, 1], total_accidents_per_weekday[i, 1])
    total_accidents_per_weekday[i, 2] = ifelse(dataset[j, i+13] == 1, total_accidents_per_weekday[i, 2] + 1, total_accidents_per_weekday[i, 2]) # update the counter
    }
}

# we first verify that the sum of total_accidents_per_weekday (column 1) is equal to the sum of the Y_vector column of the dataset
sum(total_accidents_per_weekday[, 1]) == sum(dataset$Y_vector) # TRUE

# and now we compute the average number of accidents per weekday
# by dividing the first column of total_accidents_per_weekday by the second column
# (the second column being a mere counter of how many times each weekday appears in the period 2011-2018)
average_accidents_per_weekday = total_accidents_per_weekday[, 1] / total_accidents_per_weekday[, 2] # divide the total of accidents in each weekday by the respective value of the counter 

# we can now plot the distribution of accidents over 7 weekdays
barplot(average_accidents_per_weekday, names.arg = str_sub(names(dataset[, 14:20]), 1, 3),
        main = "Average number of accidents per weekday", sub = "Average number of accidents per weekday in canton ZH over the period 2011-2018",
        xlim = c(0, 7), ylim = c(0,8))


# (4) finally we analyze the relation between precipitation amount and the number of accidents
# to do so we need data from another dataset, because Prec_amount has been standardized in this dataset
# we thus load the dataset for regression tasks that was produced in Data_wrangling.R
load("./Data/data_reg.RData") # this dataset is loaded as df_reg

# we observe that Prec_amount ranges from 2 to 477 in df_reg
min(df_reg$Prec_amount) # min = 2
max(df_reg$Prec_amount) # max = 477

# we will thus group Prec_amount in 10 buckets of 50 (from 0 to 500)

# but before doing so we need to do some data handling
# Indeed the Prec_amount values from "dataset" come from the X_matrix loaded above
# (file: covariate_matrix_reg.RData) and these values have been ranked by date
# in the scrit REGRESSION_covariate_matrix_with_dummy_variables.R
# On the contrary the Prec_amount values from "df_reg" have not been ranked by date
# (file: Data_wrangling.R)

# we thus rank "df_reg" by date
library(lubridate)
df_reg = df_reg %>% 
  mutate(date = make_date(AccidentYear, AccidentMonth, days))
df_reg = df_reg[order(df_reg$date),]

# and we verify that each Prec_amount value in "dataset" now corresponds to the standardized
# form of the respective Prec_amount value in "df_reg"
standFun = function(x){ #standFun is a function that standardizes -> mean = 0, sd = 1
  out = (x - mean(x))/sd(x)
  return(out)
}
length(unique(dataset$Prec_amount == standFun(df_reg$Prec_amount))) # length = 1, i.e. always TRUE

# now we create the 10 buckets of 50 for Prec_amount in "df_reg"
df_reg = df_reg %>%
  mutate(
    Prec_amount_0_50 = ifelse((df_reg$Prec_amount >= 0 & df_reg$Prec_amount < 50), 1, 0),
    Prec_amount_50_100 = ifelse((df_reg$Prec_amount >= 50 & df_reg$Prec_amount < 100), 1, 0),
    Prec_amount_100_150 = ifelse((df_reg$Prec_amount >= 100 & df_reg$Prec_amount < 150), 1, 0),
    Prec_amount_150_200 = ifelse((df_reg$Prec_amount >= 150 & df_reg$Prec_amount < 200), 1, 0),
    Prec_amount_200_250 = ifelse((df_reg$Prec_amount >= 200 & df_reg$Prec_amount < 250), 1, 0),
    Prec_amount_250_300 = ifelse((df_reg$Prec_amount >= 250 & df_reg$Prec_amount < 300), 1, 0),
    Prec_amount_300_350 = ifelse((df_reg$Prec_amount >= 300 & df_reg$Prec_amount < 350), 1, 0),
    Prec_amount_350_400 = ifelse((df_reg$Prec_amount >= 350 & df_reg$Prec_amount < 400), 1, 0),
    Prec_amount_400_450 = ifelse((df_reg$Prec_amount >= 400 & df_reg$Prec_amount < 450), 1, 0),
    Prec_amount_450_500 = ifelse((df_reg$Prec_amount >= 450 & df_reg$Prec_amount < 500), 1, 0))

# we verify that the sum of the newly created columns (13:22) is equal to the number of rows of df_reg
sum(df_reg[, 13:22]) == nrow(df_reg) #TRUE

# we now proceed similarly to point (3) (accidents per weekday)
# by collecting the number of accidents for each bucket of precipitation amount
total_accidents_and_precipitation = matrix(0, nrow = 10, ncol = 2) # add a second column for a counter

for(i in 1:10){
  for(j in 1:nrow(dataset)){
    total_accidents_and_precipitation[i, 1] = ifelse(df_reg[j, i+12] == 1, total_accidents_and_precipitation[i, 1] + dataset[j, 1], total_accidents_and_precipitation[i, 1])
    total_accidents_and_precipitation[i, 2] = ifelse(df_reg[j, i+12] == 1, total_accidents_and_precipitation[i, 2] + 1, total_accidents_and_precipitation[i, 2]) # update the counter
  }
}

# we verify that the sum of total_accidents_and_precipitation (column 1) is equal to the sum of the Y_vector column of the dataset
sum(total_accidents_and_precipitation[, 1]) == sum(dataset$Y_vector) # TRUE

# and now we compute the average number of accidents per "bucket" of precipitation amount
# by dividing the first column of total_accidents_and_precipitation by the second column
# (the second column being a mere counter of how many times each "bucket" of prec_amount appears in the period 2011-2018)
average_accidents_and_precipitation = total_accidents_and_precipitation[, 1] / total_accidents_and_precipitation[, 2] # divide the total of accidents in each weekday by the respective value of the counter 

# we can now plot the distribution of accidents over 10 buckets of precipitation amount
barplot(average_accidents_and_precipitation, names.arg = c("0-50", "50-100", "100-150", "150-200", "200-250", "250-300", "300-350", "350-400", "400-450", "450-500"),
        main = "Number of accidents per day and precipitation amount", sub = "Average number of accidents per day in canton ZH for a given level of precipitation amount (mm) (2011-2018)",
        ylim = c(0,10))


#### severity of accidents ####

# In this section we focus on the visualisation of the severity of accidents in CH
# Our data is split in 3 severity degrees: light injuries, severe injuries, fatalities
# We provide the following visualisation
# (1) distribution of the severity
# (2) share (percentage) of fatal accidents per month
# (3) share (percentage) of fatal accidents per weekday
# (4) correlation between precipitation amount and share (percentage) of fatal accidents


# we start by loading the data that was created for this purpose
# this data consists of "Y_vector_classification.RData" and "covariate_matrix.RData"
# both of them having been created in the script CLASSIFICATION_covariate_matrix_with_dummy_variables.R
load("./Data/Y_vector_classification.RData") # data loaded under the name "Y_vector"
load("./Data/covariate_matrix.RData") # data loaded under the name "X_matrix"
dataset = as.data.frame(cbind(Y_vector, X_matrix))


# (1) we visualise the distribution of the severity of accidents on an annual basis

# note that the Y_vector column in "dataset" represents the severity of accidents in the following way:
# 3 = light injuries; 2 = severe injuries; 1 = fatalities

# we create a 3-row column vector for doing this
distribution_severity = rep(0, 3)


for(i in 1:3){
  for(j in 1:nrow(dataset)){
    distribution_severity[i] = ifelse(dataset[j, 1] == i, distribution_severity[i] + 1, distribution_severity[i])
  }
}

# we verify that the sum of "distribution_severity" is equal to the number of rows of "dataset"
sum(distribution_severity) == nrow(dataset) # TRUE

# and now we compute the average distribution of severity on an annual basis
# by dividing each row of distribution_severity by 8
# (because there are 8 years in the period 2011-2018)
average_severity_accidents = distribution_severity / 8

# we can now plot the average severity of accidents
barplot(average_severity_accidents, names.arg = c("fatalities", "severe injuries", "light injuries"),
        main = "Annual distribution of the severity of accidents", sub = "Average annual distribution of the severity of accidents in Switzerland (2011-2018)")


# (2) we now visualise the share (percentage) of fatal accidents over 12 months
# i.e. we group the accidents by month
# and compute the share (percentage) of fatal accidents in each month
# The idea being to see whether some months are more conducive to fatal accidents than other months

total_accidents_per_month = matrix(0, nrow=12, ncol=2) # 1 column for total accidents, 1 for fatal accidents only

for(i in 1:12){
  for(j in 1:nrow(dataset)){
    total_accidents_per_month[i, 1] = ifelse(dataset[j, i+36] == 1, total_accidents_per_month[i, 1] + 1, total_accidents_per_month[i, 1])
    total_accidents_per_month[i, 2] = ifelse((dataset[j, i+36] == 1 & dataset[j, 1] == 1), total_accidents_per_month[i, 2] + 1, total_accidents_per_month[i, 2])
  }
}

# we first verify that the sum of total_accidents_per_month (column 1) is equal to the number of rows of the dataset
sum(total_accidents_per_month[, 1]) == nrow(dataset) # TRUE

# and we verify that the sum of total_accidents_per_month (column 2) is equal to the number of fatal accidents in the dataset
sum(total_accidents_per_month[, 2]) == length(which(dataset$Y_vector == 1)) # TRUE

# and now we compute the share (percentage) of fatal accidents per month
# by dividing column 2 by column 1, and multiplying the result by 100
average_accidents_per_month = 100 * (total_accidents_per_month[, 2] / total_accidents_per_month[, 1])

# we can now plot the share (percentage) of fatal accidents over 12 months
barplot(average_accidents_per_month, names.arg = str_sub(names(dataset[, 37:48]), 1, 3),
        main = "Share (percentage) of fatal accidents per month", sub = "Average share (percentage) of fatal accidents per month in Switzerland over the period 2011-2018",
        xlim = c(0,15), ylim = c(0,1.4))


# (3) we now proceed similarly to visualise the share (percentage) of fatal accidents over 7 weekdays
# i.e. we group the accidents by weekday
# and compute the share (percentage) of fatal accidents in each weekday
# The idea being to see whether some weekdays are more conducive to fatal accidents than other weekdays

total_accidents_per_weekday = matrix(0, nrow = 7, ncol = 2) # 1 column for total accidents, 1 for fatal accidents only

for(i in 1:7){
  for(j in 1:nrow(dataset)){
    total_accidents_per_weekday[i, 1] = ifelse(dataset[j, i+48] == 1, total_accidents_per_weekday[i, 1] + 1, total_accidents_per_weekday[i, 1])
    total_accidents_per_weekday[i, 2] = ifelse((dataset[j, i+48] == 1 & dataset[j, 1] == 1), total_accidents_per_weekday[i, 2] + 1, total_accidents_per_weekday[i, 2])
  }
}

# we first verify that the sum of total_accidents_per_weekday (column 1) is equal to the number of rows of the dataset
sum(total_accidents_per_weekday[, 1]) == nrow(dataset) # TRUE

# and we verify that the sum of total_accidents_per_weekday (column 2) is equal to the number of fatal accidents in the dataset
sum(total_accidents_per_weekday[, 2]) == length(which(dataset$Y_vector == 1)) # TRUE

# now we compute the share (percentage) of fatal accidents per weekday
# by dividing the second column of total_accidents_per_weekday by the first column
# and multiplying the result by 100
average_accidents_per_weekday = 100 * (total_accidents_per_weekday[, 2] / total_accidents_per_weekday[, 1])

# we can now plot the distribution of accidents over 7 weekdays
barplot(average_accidents_per_weekday, names.arg = str_sub(names(dataset[, 49:55]), 1, 3),
        main = "Share (percentage) of fatal accidents per weekday", sub = "Average share (percentage) of fatal accidents per weekday in Switzerland over the period 2011-2018",
        ylim = c(0, 1.4))

# (4) finally we analyze the relation between precipitation amount and the share (percentage) of fatal accidents
# to do so we need data from another dataset, because Prec_amount has been standardized in this dataset
# we thus load the dataset for classification tasks that was produced in Data_wrangling.R
load("./Data/data_class.RData") # this dataset is loaded as df_class

# we observe that Prec_Amount ranges from 0 to 47.1 in df_class
min(df_class$Prec_Amount) # min = 0
max(df_class$Prec_Amount) # max = 47.1

# remember that in df_reg Prec_amount ranges from 2 to 477
# this is because df_class measures Prec_Amount in cm
# whereas df_reg measures Prec_amount in mm
# for the sake of consistency we multiply df_class$Prec_Amount by 10
df_class$Prec_Amount = 10 * df_class$Prec_Amount

# we can thus group Prec_Amount in 10 buckets of 50 (from 0 to 500)
# similarly to what we did with df_reg in the previous section

# we first verify that each Prec_Amount value in "dataset" corresponds to the standardized
# form of the respective Prec_amount value in "df_class"
standFun = function(x){ #standFun is a function that standardizes -> mean = 0, sd = 1
  out = (x - mean(x))/sd(x)
  return(out)
}
length(unique(dataset$Prec_Amount == standFun(df_class$Prec_Amount / 10))) # length = 1, i.e. always TRUE (we've divided by 10 for the verification because we had multiplied by 10 above)

# now we create the 10 buckets of 50 for Prec_amount in "df_class"
df_class = df_class %>%
  mutate(
    Prec_amount_0_50 = ifelse((df_class$Prec_Amount >= 0 & df_class$Prec_Amount < 50), 1, 0),
    Prec_amount_50_100 = ifelse((df_class$Prec_Amount >= 50 & df_class$Prec_Amount < 100), 1, 0),
    Prec_amount_100_150 = ifelse((df_class$Prec_Amount >= 100 & df_class$Prec_Amount < 150), 1, 0),
    Prec_amount_150_200 = ifelse((df_class$Prec_Amount >= 150 & df_class$Prec_Amount < 200), 1, 0),
    Prec_amount_200_250 = ifelse((df_class$Prec_Amount >= 200 & df_class$Prec_Amount < 250), 1, 0),
    Prec_amount_250_300 = ifelse((df_class$Prec_Amount >= 250 & df_class$Prec_Amount < 300), 1, 0),
    Prec_amount_300_350 = ifelse((df_class$Prec_Amount >= 300 & df_class$Prec_Amount < 350), 1, 0),
    Prec_amount_350_400 = ifelse((df_class$Prec_Amount >= 350 & df_class$Prec_Amount < 400), 1, 0),
    Prec_amount_400_450 = ifelse((df_class$Prec_Amount >= 400 & df_class$Prec_Amount < 450), 1, 0),
    Prec_amount_450_500 = ifelse((df_class$Prec_Amount >= 450 & df_class$Prec_Amount < 500), 1, 0))

# we verify that the sum of the newly created columns (25:34) is equal to the number of rows of df_class
sum(df_class[, 25:34]) == nrow(df_class) #TRUE

# we now proceed similarly to point (3) (share of fatal accidents per weekday)
# by collecting the total number of accidents and the number of fatal accidents for each bucket of precipitation amount
total_accidents_and_precipitation = matrix(0, nrow = 10, ncol = 2)  # 1 column for total accidents, 1 for fatal accidents only

for(i in 1:10){
  for(j in 1:nrow(dataset)){
    total_accidents_and_precipitation[i, 1] = ifelse(df_class[j, i+24] == 1, total_accidents_and_precipitation[i, 1] + 1, total_accidents_and_precipitation[i, 1])
    total_accidents_and_precipitation[i, 2] = ifelse((df_class[j, i+24] == 1 & dataset[j, 1] == 1), total_accidents_and_precipitation[i, 2] + 1, total_accidents_and_precipitation[i, 2])
  }
}

# we verify that the sum of total_accidents_and_precipitation (column 1) is equal to the number of rows of the dataset
sum(total_accidents_and_precipitation[, 1]) == nrow(dataset) # TRUE

# and we verify that the sum of total_accidents_and_precipitation (column 2) is equal to the number of fatal accidents in the dataset
sum(total_accidents_and_precipitation[, 2]) == length(which(dataset$Y_vector == 1)) # TRUE

# now we compute the share (percentage) of fatal accidents per "bucket" of precipitation amount
# by dividing the second column of total_accidents_and_precipitation by the first column
# and multiplying the result by 100
average_accidents_and_precipitation = 100 * (total_accidents_and_precipitation[, 2] / total_accidents_and_precipitation[, 1])

# we can now plot the share (percentage) of fatal accidents over 10 buckets of precipitation amount
barplot(average_accidents_and_precipitation, names.arg = c("0-50", "50-100", "100-150", "150-200", "200-250", "250-300", "300-350", "350-400", "400-450", "450-500"),
        main = "Share (percentage) of fatal accidents per precipitation amount", sub = "Average share (percentage) of fatal accidents in Switzerland for a given level of precipitation amount (mm) (2011-2018)")


