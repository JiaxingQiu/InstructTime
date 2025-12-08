rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../../")
print(getwd())

# --- load packages ---
library(readxl)
library(ggplot2)
library(reshape2)

# --- load functions ---
path = paste0("./script/Data/utils")
flst = list.files(path)
sapply(c(paste(path,flst,sep="/")), source, .GlobalEnv)


# --- load 10-minutes records ---
ts_hr <- read_excel("./data/PAS Challenge HR Data.xlsx")
ts_sp <- read_excel("./data/PAS Challenge SPO2 Data.xlsx")
view_k_row(ts_hr) # viz first 10 rows
view_k_row(ts_sp)
ts_sub <- ts_hr[,3:ncol(ts_hr)]
# events80 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 80, plot=F, type = 0))
# events90 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 90, plot=F, type = 0))
# events100 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 100, plot=F, type = 0))
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
# ---- descriptions ----
df_desc <- ts_hr[,c(1:2)]
succ <- unlist(apply(ts_sub, 1, function(x) describe_succ_inc_summ(x)))
df_desc$description_succ_inc <- succ
table(df_desc$description_succ_inc)
histo <- unlist(apply(ts_sub, 1, function(x) describe_hr_histogram(x)))
df_desc$description_histogram <- histo
table(df_desc$description_histogram)
events80 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 80, plot=F, type = 0))
events90 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 90, plot=F, type = 0))
events100 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 100, plot=F, type = 0))
most_severe_events <- sapply(seq(1,nrow(ts_sub)), get_most_severe_event)
row_index <- which(!is.na(most_severe_events))
df_desc$description_ts_event <- ""
df_desc$description_ts_event[row_index] <- most_severe_events[row_index]
write.csv(df_desc, "./data/hr_descriptions.csv", row.names = F)

# 
# # ----- augmented ------
# ts_hr_aug <- read.csv("./data/hr_aug.csv")
# ts_sub <- ts_hr_aug[,c(2:301)]
# colnames(ts_sub) <- seq(1,300,1)
# view_k_row(ts_sub)
# events80 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 80, plot=F, type = 0))
# events90 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 90, plot=F, type = 0))
# events100 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 100, plot=F, type = 0))
# most_severe_events <- sapply(seq(1,nrow(ts_sub)), get_most_severe_event)
# row_index <- which(!is.na(most_severe_events))
# ts_event <- ts_hr_aug[row_index, ]
# ts_event$event_description <- most_severe_events[row_index]
# write.csv(ts_event, "./data/hr_aug.csv", row.names = F)
# 
# # ------ test ------
# ts_hr_test <- read_excel("./data/Test Data/Test HR Data.xlsx")
# ts_sub <- ts_hr_test[,3:(ncol(ts_hr_test))]
# colnames(ts_sub) <- seq(1,300,1)
# view_k_row(ts_sub)
# events80 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 80, plot=F, type = 0))
# events90 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 90, plot=F, type = 0))
# events100 <- apply(ts_sub, 1, function(x) describe_brady_event(x, th = 100, plot=F, type = 0))
# most_severe_events <- sapply(seq(1,nrow(ts_sub)), get_most_severe_event)
# row_index <- which(!is.na(most_severe_events))
# ts_event <- ts_hr_test[row_index, ]
# ts_event$event_description <- most_severe_events[row_index]
# write.csv(ts_event, "./data/Test Data/hr_test.csv", row.names = F)
# 
