library(dplyr)

#Choose directory:
setwd("/media/shareddata/MIT/Capstone")

getwd()

data <- read.csv("data1.csv", sep=",", stringsAsFactors = FALSE)
data2 <- read.csv("data2.csv", sep=",", stringsAsFactors = FALSE)


data %>%
  group_by(Carrier, ConsigneeName, Consignee, Shipper) %>%
  summarize(N = n())


names(data)
distinct(data, Carrier)
distinct(data, Original.Port.Of.Loading)
distinct(data, Original.Port.Of.Loading.Site)
distinct(data, Final.Port.Of.Discharge)
distinct(data, Final.Port.Of.Discharge.Site)


names(data2)
distinct(data2, Carrier)
distinct(data2, Original.Port.Of.Loading)
distinct(data2, Original.Port.Of.Loading.Site)
distinct(data2, Final.Port.Of.Discharge)
distinct(data2, Final.Port.Of.Discharge.Site)



class(data$Receipt.Date)


data$Receipt.Date <- as.Date(data$Receipt.Date, "%m/%d/%Y")
data2$Receipt.Date <- as.Date(data2$Receipt.Date, "%m/%d/%Y")

?as.Date
min(data$Receipt.Date)
min(data2$Receipt.Date)
max(data$Receipt.Date)
max(data2$Receipt.Date)


names(data)

summary(data)
dates <- c(names(data[1:10]), names(data[27:30]), names(data[33:37]))

for (date in dates){
  data[date] <- as.Date(data[date], "%m/%d/%Y")  
}
for (date in dates){
  paste(date)
}
