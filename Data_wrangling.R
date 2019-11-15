rm(list=ls())

###################
library(tidyverse)

# Let's start with reading the data, doing some quality 
# of life changes, and saving it as RData

accidents <- read.csv("./Data/RoadTrafficAccidentLocations.csv")

accidents <- accidents %>% 
  select(-AccidentType_de, -AccidentType_fr, -AccidentType_it,
         -RoadType_de, -RoadType_fr, -RoadType_it,
         -AccidentSeverityCategory_de, -AccidentSeverityCategory_fr, -AccidentSeverityCategory_it,
         -AccidentMonth_de, -AccidentMonth_fr, -AccidentMonth_it,
         -AccidentWeekDay_de, -AccidentWeekDay_fr, -AccidentWeekDay_it)

# now for the wheater files

wheater1 <- read.csv("./Data/9159828051199dat.txt", skip = 2) # sadly the column names are a weird combination of the first and second line, so we have to hard code them
names <- c("Name", "USAF", "NCDC ", "Date", "HrMn", "I", "Type", "QCP", "Temp", "Q1", "Dewpoint", "Q2", "Pressure", "Q3", "Relative_Humidity", "Null")
names(wheater1) <- names

# The colums denoted with an I or Q are just non-relevant quality and type indicator, we drop them togheter with 
# the empty 16th column, QCP and NCDC, which are also empty. THe type of observation is also irrelevant.

wheater1 <- wheater1 %>% 
  select(- `NCDC `  , -Type, -I, -Q1, -Q2, -Q3, -QCP, -Null)

wheater2 <- read.csv("./Data/938008051210dat.txt", skip = 2)
names2 <- c("Name", "USAF", "NCDC", "Date", "HrMn", "I", "Type",  "QCP",  "Wind_Dir", "Q1", "I2", "Wind_Spd", "Q2", "Height_of_clouds", "Q3", "I3", "I4", "Visible_distance",  "Q4", "I5", "Q6", "Null")
names(wheater2) <- names2

wheater2 <- wheater2 %>%
  select(Name, USAF, Date, HrMn, Wind_Spd, Height_of_clouds, Visible_distance)

wheater3 <- read.csv("./Data/5555778051215dat.txt", skip = 2)
names3 <- c("Name", "USAF", "NCDC", "Date", "HrMn", "I", "Type", "QCP", "Period", "Prec_Amount", "I2", "Q", "Period2", "Prec_Amount2", "I3", "Q2", "Period3", "Prec_Amount3", "I4", "Q3", "Period4", "Prec_Amount4", "I5", "Q4", "Null")
names(wheater3) <- names3

wheater3 <- wheater3 %>% 
  select(Name, USAF, Date, HrMn, Prec_Amount)

wheater4 <- read.csv("./Data/7670498051219dat.txt", skip = 2)
names4 <- c("Name", "USAF", "NCDC","Date", "HrMn", "I", "Type", "QCP", "Period", "Snow_Amount", "I2", "Q", "Pr", "Amt", "I3", "Q2", "Pr2", "Amt2", "I4", "Q3", "Pr3", "Amt3", "I5", "Q4", "Null")
names(wheater4) <- names4

wheater4 <- wheater4 %>% 
  select(Name, USAF, Date, HrMn, Snow_Amount)

wheater_stations <- read.csv("./Data/9159828051199stn.txt", sep = "") # This is a dataset of all the stations, it'll be useful since it contains the spatial coordinates of the stations

wheater_stations <- wheater_stations %>% 
  select(USAF.WBAN_ID, LATITUDE, LONGITUDE)

names_stations <- c("USAF", "LAT", "LONG")
names(wheater_stations) <- names_stations
# now we can join the wheater datasets

wheater12 <- merge(wheater1, wheater2, by = c("Name", "USAF", "Date", "HrMn"), all.x = T)   # we want to keep all observations, so
wheater123 <- merge(wheater12, wheater3, by = c("Name", "USAF", "Date", "HrMn"), all.x = T)  # we set NA where we lack obervations
wheater1234 <- merge(wheater123, wheater4, by = c("Name", "USAF", "Date", "HrMn"), all.x = T)

wheater_final <- merge(wheater1234, wheater_stations, by = "USAF", all.x = T)

names(wheater_final)
sum(is.na(wheater_final$USAF)) # all stations have been found


# we now have 2 datasets with (hopefully) enough info for a model
# let's save them

save(accidents, file = "Data/accidents.RData")
save(wheater_final, file = "Data/wheater.RData")

#################################################################################################################################

# Let's start tidyng the accident dataset up

rm(list= ls())  # environment was really full

load("./Data/accidents.RData")

names(accidents) # we still have some columns to remove

accidents <- accidents %>% 
  select(-AccidentType, -RoadType, -AccidentMonth_en, -AccidentWeekDay, - AccidentHour_text)

names(accidents)

unique(accidents$AccidentWeekDay_en) # we want to assign numbers from 1 to 7

new_levels <- c("Sunday", "Saturday", "Friday", "Thursday", "Wednesday", "Tuesday", "Monday")
for (f in new_levels){
  accidents$AccidentWeekDay_en <- relevel(accidents$AccidentWeekDay_en, f)
}
names(accidents)[15]<-paste("week_day")
accidents <- accidents %>% 
  mutate("week_day_number" = as.integer(week_day))
accidents <- subset(accidents, select = c(1:15, week_day_number, 16))

names(accidents)

accidents <- accidents %>% 
  mutate("road_type" = as.integer(RoadType_en))

accidents <- subset(accidents, select = c(1:8, road_type, 9:17))

accidents <- accidents %>% 
  mutate("fatalties" = ifelse(AccidentSeverityCategory == "as1", 1, 0)) %>% 
  mutate("severe_injuries" = ifelse(AccidentSeverityCategory == "as2", 1, 0)) %>% 
  mutate("light_injuries" = ifelse(AccidentSeverityCategory == "as3", 1, 0)) %>% 
  subset(select = c(1:4, fatalties, severe_injuries, light_injuries, 5:18))

accidents <- accidents %>% 
  mutate("Canton" = as.integer(CantonCode)) %>% 
  subset(select = c(1:15, Canton, 16:21))

# to join it with the wheater dataset, we need to change the spatial coordinates into the global system,
# this requires us some computations

get_latitude_longitude = function(CHLV95_E, CHLV95_N){
  
  # first we specify the coordinates of the Zimmerwald station (reference point of the CHLV95 system)
  E_zero = 2600000 # CHLV95_E coordinate of the reference point
  N_zero = 1200000 # CHLV95_N coordinate of the reference point
  
  # then transform the coordinates into the form required for calculations
  Y = (CHLV95_E - E_zero)/(10**6)
  X = (CHLV95_N - N_zero)/(10**6)
  # we divide by 10**6 because the calculations that follow are based on a different unit (1000km) than the unit of the function's inputs (m)
  
  # then compute an intermediate value (lambda) for the longitude (unit: 10000 seconds (i.e. 10000''))
  a_1 = 4.72973056 + 0.7925714 * X + 0.132812 * (X^2) + 0.02550 * (X^3) + 0.0048 * (X^4)
  a_3 = - 0.044270 - 0.02550 * X - 0.0096 * (X^2)
  a_5 = 0.00096
  lambda = 2.67825 + a_1*Y + a_3*(Y^3) + a_5*(Y^5)
  
  # then compute an intermediate value (phi) for the latitude (unit: 10000 seconds (i.e. 10000''))
  p_0 = 3.23864877 * X - 0.0025486 * (X^2) - 0.013245 * (X^3) + 0.000048 * (X^4)
  p_2 = - 0.27135379 - 0.0450442 * X - 0.007553 * (X^2) - 0.00146 * (X^3)
  p_4 = 0.002442 + 0.00132 * X
  phi = 16.902866 + p_0 + p_2*(Y^2) + p_4*(Y^4)
  
  # then convert phi and lambda into latitude and longitude (unit: degrees)
  longitude = lambda * 10000 / 3600
  latitude = phi * 10000 / 3600
  # first multiply by 10000 because phi and lambda are expressed in 10000 seconds
  # then divide by 3600 because 1 degree = 3600 seconds
  
  # finally return a matrix with both values
  return(cbind(longitude, latitude))
}

names(accidents)
coordinates <- accidents[,13:14]

accidents <- accidents %>%
  mutate("Longitude" = get_latitude_longitude( coordinates[,1], coordinates[,2])[,1]) %>% 
  mutate("Latitude"  = get_latitude_longitude( coordinates[,1], coordinates[,2])[,2])

accidents <- subset(accidents, select = c(1:12, Latitude, Longitude, 15:22))

# another requirement is to match the dates, sadly the accident dataset never contained the number of the day
# as of the calendar, but only the week day, so we need to find the number of the day for every month and year

days <- rep(NA, nrow(accidents))
days[1] <- 1
sunset <- 1

for (i in 2:nrow(accidents)){
  if (accidents$week_day[i] == accidents$week_day[i-1]){
    days[i] <- sunset
  }
  else {
    if (accidents$AccidentMonth[i] != accidents$AccidentMonth[i-1]){
      sunset <- 1
      days[i] <- 1
    } else {
      sunset <- sunset + 1
      days[i] <- sunset
    }
  }
}

accidents <- accidents %>% 
  mutate("days" = days)

# This would normally not work, but we were lucky enough to have 
# an accident happening at least once every day from 01.01.2011 to 31.12.2018

# Let's clean the wheater data now

load("./Data/wheater.RData")

# we want the hour to be the same format as in the accidents dataset

wheater_final <- wheater_final %>% 
  mutate("AccidentHour" = HrMn/100)

# we want to have a column for the year, one for the month, and one for the day

wheater_final <- wheater_final %>% 
  mutate("AccidentYear" = as.integer(str_sub(wheater_final$Date, 1, 4))) %>% 
  mutate("AccidentMonth" = as.integer(str_sub(wheater_final$Date, 5, 6))) %>% 
  mutate("days" = as.integer(str_sub(wheater_final$Date, 7, 8)))

# merging by coordinates

library(sp)
library(raster)

sp_acc <- SpatialPoints(cbind(accidents$Longitude, accidents$Latitude))
spdf_acc <- SpatialPointsDataFrame(sp_acc, data.frame(1:nrow(accidents)))

wheater_sel <- wheater_final %>% 
  distinct(Name, LAT, LONG)

sp_whe <- SpatialPoints(cbind(wheater_sel$LONG, wheater_sel$LAT))
spdf_whe <- SpatialPointsDataFrame(sp_whe, data.frame(1:nrow(wheater_sel)))

dist_mat <- pointDistance(spdf_acc, spdf_whe, lonlat = FALSE, allpairs = TRUE)

nearest <- apply(dist_mat, 1, which.min)

accidents <- cbind(accidents, wheater_sel[nearest,])

merged <- inner_join(accidents, wheater_final, by = c("Name", "AccidentYear", "AccidentMonth", "days", "AccidentHour"))

# Let's do some cleaning

names(merged)

# Many variables are useless for modelling and sadly, some variables are mostly NA, so we drop them

df <- merged %>% 
  dplyr::select(AccidentSeverityCategory, AccidentSeverityCategory_en, fatalties, severe_injuries, light_injuries,
          AccidentInvolvingPedestrian, AccidentInvolvingBicycle, AccidentInvolvingMotorcycle, RoadType_en, road_type,
          CantonCode, Canton, AccidentYear, AccidentMonth, week_day, week_day_number, days, AccidentHour, Temp, Dewpoint, Pressure,
          Relative_Humidity, Wind_Spd, Prec_Amount)

# There're also some NA in Prec_Amount, but we want to keep them for visualization
# more problematic are the "NAs" introduced by the website from which we took the data: They're a bunch of "9", 
# so they would disrupt he data visualization

df <- df[which(df$Prec_Amount != 999.9&df$Pressure != 9999.9),]

df_vis <- df
save(df_vis, file = "Data/data_vis.RData")

# now let's make a cleaner dataset for modeling

df_class <- na.omit(df)

save(df_class, file = "Data/data_class.RData")

### Let's get a dataset to regress on.
### We want to know the number of accidents in the canton ZH in a given day!

rm(list=ls())

load("./Data/tidy_dataset")
wheater_zh <- read.csv("./Data/meteo_zh.txt", skip = 2, sep = "")

colnames(wheater_zh) <- c("station", "date", "Pressure", "Temp", "Humidity", "Prec1", "Prec2", "Prec_total")

wheater_zh <- wheater_zh[which(wheater_zh$date >= "20110101" & wheater_zh$date < 20190101),]

wheater_zh <- wheater_zh %>% 
  mutate("AccidentYear" = as.integer(str_sub(wheater_zh$date, 1, 4))) %>% 
  mutate("AccidentMonth" = as.integer(str_sub(wheater_zh$date, 5, 6))) %>% 
  mutate("days" = as.integer(str_sub(wheater_zh$date, 7, 8)))

wheater_zh <- wheater_zh %>%
  mutate("Prec_amount" = as.double(Prec1) + as.double(Prec2)) %>% 
  dplyr::select(AccidentYear, AccidentMonth, days, Temp, Pressure, Humidity, Prec_amount)


df_reg <- df %>% 
  dplyr::select(CantonCode, AccidentYear, AccidentMonth, week_day, week_day_number, days)

df_reg <- df_reg %>% 
  mutate(date = sprintf("%s%s%s", days, AccidentMonth, AccidentYear))

df_reg_zh <- df_reg[which(df_reg$CantonCode == "ZH"),]

sum_acc <- as_tibble(df_reg_zh[1,])
sum_acc <- sum_acc %>% 
  mutate(number_accidents = 1)

df_reg_zh <- df_reg_zh %>% 
  mutate(number_accidents = 1)

j <- 1

for (i in 2:nrow(df_reg_zh)){
  
  if (df_reg_zh$date[i] == df_reg_zh$date[i-1]){
    
    sum_acc$number_accidents[j] <- sum_acc$number_accidents[j] + 1
    
  } else{
    
    sum_acc <- rbind(sum_acc, df_reg_zh[i,])
    j = j + 1
    
  }
}

df_reg_zh <- sum_acc

df_reg <- merge(wheater_zh, df_reg_zh, by = c("AccidentYear", "AccidentMonth", "days"), all.x = T)

df_reg <- df_reg %>% 
  dplyr::select(CantonCode, AccidentYear, AccidentMonth, days, week_day_number, Temp, Pressure, Humidity, Prec_amount, number_accidents)

df_reg$CantonCode[is.na(df_reg$CantonCode)] <- "ZH"
df_reg$number_accidents[is.na(df_reg$number_accidents)] <- 0

for (i in 1:nrow(df_reg)){
  
  if (is.na(df_reg$week_day_number[i]) == T){
    df_reg$week_day_number[i] <- df_reg$week_day_number[i-1] + 1
  }
}

# we also want the wind-speed

zh_wind <- read.csv("./Data/wind_zh.txt", skip = 2)
colnames(zh_wind) <- c("Name", "USAF", "NCDC", "Date", "HrMn", "I", "Type", "QCP", "Dir", "Q", "I2", "Wind_Spd", "Q2", "Null")
zh_wind <- zh_wind %>% 
  dplyr::select(Date, Wind_Spd)

zh_wind <- zh_wind %>% 
  mutate("AccidentYear" = as.integer(str_sub(zh_wind$Date, 1, 4))) %>% 
  mutate("AccidentMonth" = as.integer(str_sub(zh_wind$Date, 5, 6))) %>% 
  mutate("days" = as.integer(str_sub(zh_wind$Date, 7, 8))) %>% 
  dplyr::select(-Date)

save(df_reg, file = "Data/data_reg.RData")
