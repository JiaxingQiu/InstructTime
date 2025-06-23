rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../../")
print(getwd())

# --- load packages ---
library(readxl)
library(ggplot2)
library(reshape2)
library(zoo)
library(dplyr)
library(RcppRoll)

# --- load functions ---
path = paste0("./script/Data/utils")
flst = list.files(path)
sapply(c(paste(path,flst,sep="/")), source, .GlobalEnv)


# --- load 10-minutes records ---
ts_hr_org <- read_excel("./data/nicu/PAS Challenge HR Data.xlsx")
# ts_sp <- read_excel("./data/nicu/PAS Challenge SPO2 Data.xlsx")

ts_hr = ts_hr_org
view_k_row(ts_hr) # viz first 10 rows
original_data <- ts_hr %>%
  select(`1`:`300`) %>%
  mutate_all(as.numeric) %>%
  as.matrix()

smoothed_data <- t(roll_mean(t(original_data), n = 3, fill = "extend", align = "center"))
smoothed_data[, 1] <- smoothed_data[, 2]
smoothed_data[, ncol(smoothed_data)] <- smoothed_data[, ncol(smoothed_data) - 1]
ts_hr[, as.character(1:300)] <- smoothed_data
view_k_row(ts_hr) # viz first 10 rows
rm(original_data, smoothed_data)
ts_sub <- ts_hr %>% select(`1`:`300`) %>% as.data.frame()







histo = apply(ts_sub, 1, function(x) describe_hr_histogram(x))
table(histo)
view_k_row(ts_hr[histo %in% c("High variability."),]) 
view_k_row(ts_hr[histo %in% c("Moderate variability."),]) 
view_k_row(ts_hr[histo %in% c("Low variability."),]) 

succ <- apply(ts_sub, 1, function(x) describe_succ_unc_summ(x))
succ <- unlist(succ)
table(succ)
succ <- apply(ts_sub, 1, function(x) describe_succ_inc_summ(x))
succ <- unlist(succ)
table(succ)
view_k_row(ts_hr[succ %in% c("Low amount of consecutive increases."),]) 
view_k_row(ts_hr[succ %in% c("Moderate amount of consecutive increases."),]) 
view_k_row(ts_hr[succ %in% c("High amount of consecutive increases."),]) 

events80 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 80, plot=F, type = 0))
events90 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 90, plot=F, type = 0))
events100 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 100, plot=F, type = 0))
# # apply the function to each row_index of ts_sub
# most_severe_events <- sapply(seq(1,nrow(ts_sub)), get_most_severe_event)
# # count NA in most_severe_events
# sum(!is.na(most_severe_events))
# # get row index of not NA in most_severe_events
# row_index <- which(!is.na(most_severe_events))
# # ts_event dataframe
# ts_event <- ts_hr[row_index, ]
# # get values of most_severe_events for the row_index
# ts_event_description <- most_severe_events[row_index]
# ts_event$event_description <- ts_event_description
# write.csv(ts_event, "./data/HR_events.csv", row.names = F)
