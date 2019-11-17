#### Introduction ####
# In this script we create a covariate matrix specifically for classication tasks
# The idea is to convert all categorical variables (e.g. road type, canton) into dummy variables (1 or 0)
# In doing so we avoid weigthing problems
# e.g. we avoid that the model gives too much weight to the variable "CantonCode" simply because this variable ranges from 1 to 26


rm(list=ls())
library(tidyverse)


# we load the dataset for classification tasks that was produced in Data_wrangling.R
load("./Data/data_class.RData")


#### X_matrix ####
# we create the X_matrix of covariates

# we first drop the target columns (Y)
X_matrix = df_class %>%
  dplyr::select(-c(AccidentSeverityCategory, AccidentSeverityCategory_en))

# we then convert the true/false values of 3 columns into dummy variables (1 or 0)
# these columns are AccidentInvolvingPedestrian, AccidentInvolvingBicycle, AccidentInvolvingMotorcycle
X_matrix = X_matrix %>%
  mutate(AccidentInvolvingPedestrian = as.integer(AccidentInvolvingPedestrian)-1,
         AccidentInvolvingBicycle = as.integer(AccidentInvolvingBicycle)-1,
         AccidentInvolvingMotorcycle = as.integer(AccidentInvolvingMotorcycle)-1)

# we then convert the road type values into dummy variables (1 or 0)
X_matrix = X_matrix %>%
  mutate(Expressway = ifelse(road_type == 1, 1, 0),
         Minor_road = ifelse(road_type == 2, 1, 0),
         Motorway = ifelse(road_type == 3, 1, 0),
         Motorway_side_installation = ifelse(road_type == 4, 1, 0),
         Other = ifelse(road_type == 5, 1, 0),
         Principal_road = ifelse(road_type == 6, 1, 0)) %>%
  dplyr::select(1:6, 23:28, 9:22)

# we then convert the canton codes into dummy variables (1 or 0)
X_matrix = X_matrix %>%
  mutate(AG = ifelse(Canton == 1, 1, 0),
         AI = ifelse(Canton == 2, 1, 0),
         AR = ifelse(Canton == 3, 1, 0),
         BE = ifelse(Canton == 4, 1, 0),
         BL = ifelse(Canton == 5, 1, 0),
         BS = ifelse(Canton == 6, 1, 0),
         FR = ifelse(Canton == 7, 1, 0),
         GE = ifelse(Canton == 8, 1, 0),
         GL = ifelse(Canton == 9, 1, 0),
         GR = ifelse(Canton == 10, 1, 0),
         JU = ifelse(Canton == 11, 1, 0),
         LU = ifelse(Canton == 12, 1, 0),
         NE = ifelse(Canton == 13, 1, 0),
         NW = ifelse(Canton == 14, 1, 0),
         OW = ifelse(Canton == 15, 1, 0),
         SG = ifelse(Canton == 16, 1, 0),
         SH = ifelse(Canton == 17, 1, 0),
         SO = ifelse(Canton == 18, 1, 0),
         SZ = ifelse(Canton == 19, 1, 0),
         TG = ifelse(Canton == 20, 1, 0),
         TI = ifelse(Canton == 21, 1, 0),
         UR = ifelse(Canton == 22, 1, 0),
         VD = ifelse(Canton == 23, 1, 0),
         VS = ifelse(Canton == 24, 1, 0),
         ZG = ifelse(Canton == 25, 1, 0),
         ZH = ifelse(Canton == 26, 1, 0)) %>%
  dplyr::select(1:12, 27:52, 15:26)

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
  dplyr::select(1:38, 50:61, 40:49)

# we then convert the weekdays into dummy variables (1 or 0)
X_matrix = X_matrix %>%
  mutate(Monday = ifelse(week_day_number == 1, 1, 0),
         Tuesday = ifelse(week_day_number == 2, 1, 0),
         Wednesday = ifelse(week_day_number == 3, 1, 0),
         Thursday = ifelse(week_day_number == 4, 1, 0),
         Friday = ifelse(week_day_number == 5, 1, 0),
         Saturday = ifelse(week_day_number == 6, 1, 0),
         Sunday = ifelse(week_day_number == 7, 1, 0)) %>%
  dplyr::select(1:50, 61:67, 53:60)

# we then drop the column days, because it is not relevant for our predictions
# indeed what matter is the month and the weekday, but not the day number in the month
X_matrix = X_matrix %>% 
  dplyr::select(-days)

# we then convert the accident hour into dummy variables (1 or 0)
X_matrix = X_matrix %>%
  mutate(Hour0 = ifelse(AccidentHour == 0, 1, 0),
         Hour1 = ifelse(AccidentHour == 1, 1, 0),
         Hour2 = ifelse(AccidentHour == 2, 1, 0),
         Hour3 = ifelse(AccidentHour == 3, 1, 0),
         Hour4 = ifelse(AccidentHour == 4, 1, 0),
         Hour5 = ifelse(AccidentHour == 5, 1, 0),
         Hour6 = ifelse(AccidentHour == 6, 1, 0),
         Hour7 = ifelse(AccidentHour == 7, 1, 0),
         Hour8 = ifelse(AccidentHour == 8, 1, 0),
         Hour9 = ifelse(AccidentHour == 9, 1, 0),
         Hour10 = ifelse(AccidentHour == 10, 1, 0),
         Hour11 = ifelse(AccidentHour == 11, 1, 0),
         Hour12 = ifelse(AccidentHour == 12, 1, 0),
         Hour13 = ifelse(AccidentHour == 13, 1, 0),
         Hour14 = ifelse(AccidentHour == 14, 1, 0),
         Hour15 = ifelse(AccidentHour == 15, 1, 0),
         Hour16 = ifelse(AccidentHour == 16, 1, 0),
         Hour17 = ifelse(AccidentHour == 17, 1, 0),
         Hour18 = ifelse(AccidentHour == 18, 1, 0),
         Hour19 = ifelse(AccidentHour == 19, 1, 0),
         Hour20 = ifelse(AccidentHour == 20, 1, 0),
         Hour21 = ifelse(AccidentHour == 21, 1, 0),
         Hour22 = ifelse(AccidentHour == 22, 1, 0),
         Hour23 = ifelse(AccidentHour == 23, 1, 0)) %>%
  dplyr::select(1:57, 65:88, 59:64)

# then we omit observations where Pressure = 9999.9 or Prec_Amount = 999.9
# because they correspond to NA values
X_matrix = X_matrix[which(X_matrix$Prec_Amount != 999.9 & X_matrix$Pressure != 9999.9),]

# finally we convert the X_matrix (currently a dataframe) into a matrix
X_matrix = data.matrix(X_matrix)
