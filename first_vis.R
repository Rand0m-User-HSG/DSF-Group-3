rm(list = ls())
library(tidyverse)

load("./data/data_reg.RData")
df_wheater <- df_reg %>% 
  select(number_accidents, Temp, Pressure, Humidity, Wind_Spd, Prec_amount)

res <- cor(df_wheater)
round(res, 2)
cor(df_wheater, use = "complete.obs")

install.packages("corrplot")
library(corrplot)
corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

ggplot(df_vis)+
  geom_histogram(aes(x = AccidentHour, fill = AccidentSeverityCategory), binwidth = 1)

ggplot(df_vis)+
  geom_histogram(aes(x = week_day_number, fill = AccidentSeverityCategory), binwidth = 1)
