rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../../")
print(getwd())

# --- load packages ---
library(readxl)
library(ggplot2)
library(reshape2)
library(dplyr)

# --- load functions ---
path = paste0("./script/Data/utils")
flst = list.files(path)
sapply(c(paste(path,flst,sep="/")), source, .GlobalEnv)


# --- load uu / uuu records ---
ts <- readr::read_csv("./script/CLIP/tmp/df_train.csv.zip") 
succ <- apply(ts[,4:ncol(ts)], 1, function(x) successive_increases(x))
df_succ <- do.call(rbind, lapply(succ, function(s) data.frame(uu = s$uu, uuu = s$uuu)))
df_train <- bind_cols(ts[,c(1:3)], df_succ)
df_train$y <- ifelse(df_train$cl_event=="This infant will survive.", 0,1)
df_train$umean = df_train$uu * 2 + df_train$uuu * 3
df_train_neg <- df_train[which(df_train$y==0),]
df_train_neg <- df_train_neg[sample(1:nrow(df_train_neg), 1000),]
df_train_mdl <- rbind(df_train[which(df_train$y==1),], df_train_neg)


ts <- readr::read_csv("./script/CLIP/tmp/df_test.csv.zip") 
succ <- apply(ts[,4:ncol(ts)], 1, function(x) successive_increases(x))
df_succ <- do.call(rbind, lapply(succ, function(s) data.frame(uu = s$uu, uuu = s$uuu)))
df_test <- bind_cols(ts[,c(1:3)], df_succ)
df_test$y <- ifelse(df_test$cl_event=="This infant will survive.", 0,1)
df_test$umean = df_test$uu * 2 + df_test$uuu * 3
# downsample 1000 from df_test$y == 0, keep all df_test$y == 1
df_test_neg <- df_test[which(df_test$y==0),]
df_test_neg <- df_test_neg[sample(1:nrow(df_test_neg), 1000),]
df_test_mdl <- rbind(df_test[which(df_test$y==1),], df_test_neg)

# ---- logistic regression --- 
mdl <- glm(y~umean, data = df_train_mdl, family = "binomial")
# predict in probability
df_train_mdl$y_pred <- predict(mdl, newdata = df_train_mdl, type="response") 
df_test_mdl$y_pred <- predict(mdl, newdata = df_test_mdl, type="response") 

# --- calculate AUC --- 
library(pROC)
auc(roc_obj <- roc(df_train_mdl$y, df_train_mdl$y_pred))
auc(roc_obj <- roc(df_test_mdl$y, df_test_mdl$y_pred)) # 0.78 - 0.79


