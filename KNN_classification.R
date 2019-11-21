##### KNN Classifier ####
# In sthis script we will apply KNN classification. 
# We want to classifiy the severity of accidents.
# Beforehand we cleaned the data into three categories: 
# Accident with light injuries (as3), Accident with severe injuries	(as2), Accident with fatalities	(as1).


rm(list=ls())
library(tidyverse)
library(dplyr)

# We load the Data for classification.
load("~/Desktop/HSG/DSF/Group Work/Data/data_reg.RData")


# We delete columns in case two or more columns correlate. 
# For example we will delete "Road_Type_en" because it correlates with "road_type".
df_class <- df_class %>%
  dplyr::select(-c(AccidentSeverityCategory_en, RoadType_en, CantonCode, week_day))

# We normalized the columns in order to get zeros and ones.


# We use this library for KNN.
library(class)

# We sample into training and data set.
set.seed(123)
data_sampled <- sample(1:nrow(X_matrix),size=nrow(X_matrix)*0.7,replace = FALSE) #random selection of 70% data.

training_data <- X_matrix[data_sampled,] # 70% training data
testing_data <- X_matrix[-data_sampled,] #  30% test data



#Creating seperate dataframe for 'Creditability' feature which is our target.
training_data_labels <- X_matrix[data_sampled,1]
testing_data_labels <-X_matrix[-data_sampled,1]


#Find the number of observation
NROW(training_data) 
# 38003
sqrt(38003)
# ca. 190


# We will test three different K-values: K = 5, K = 190, K = 250. 
knn.5 <- knn(train=training_data, test=testing_data, cl=training_data_labels, k=5)
knn.190 <- knn(train=training_data, test=testing_data, cl=training_data_labels, k=190)
knn.250 <- knn(train=training_data, test=testing_data, cl=training_data_labels, k=250)


#We Calculate the proportion of correct classification for k = 5, 190, 250
Accuracy.5 = 100 * sum(testing_data_labels == knn.5)/NROW(testing_data_labels)
Accuracy.190 <- 100 * sum(testing_data_labels == knn.190)/NROW(testing_data_labels)
Accuracy.250 <- 100 * sum(testing_data_labels == knn.250)/NROW(testing_data_labels)
#Accuracy.5
#86.4
#Accuracy.190
#85.7
#Accuracy.250
#85.7


# Check prediction against actual value in tabular form for k=5
table(knn.5 ,testing_data_labels)

# Check prediction against actual value in tabular form for k=190
table(knn.190 ,testing_data_labels)

# Check prediction against actual value in tabular form for k=250
table(knn.250 ,testing_data_labels)




