successive_unchanges <- function(y) {
  # Input: y - numeric vector of heart rate measurements
  # Output: list containing counts and proportions of successive unchanges
  
  # Convert to binary: 1 for increase, 0 for same, -1 for decrease
  N <- length(y)
  if (N < 5) {
    warning("Time series too short")
    return(NULL)
  }
  
  # Get binary sequence (1 for increase, 0 for decrease/same)
  y_diff <- diff(y)  # Get differences
  y_bin <- as.numeric(y_diff == 0)  # 1 for increase, 0 for same, -1 for decrease
  
  # Initialize output list
  out <- list()
  
  # Single unchanges (length 1)
  out$u <- mean(y_bin == 1)  # proportion of unchanges
  
  # Two consecutive unchanges (length 2)
  n2 <- length(y_bin) - 1
  out$uu <- mean(y_bin[1:n2] == 1 & y_bin[2:(n2+1)] == 1)  # unchange,unchange
  
  # Three consecutive (length 3)
  n3 <- length(y_bin) - 2
  out$uuu <- mean(y_bin[1:n3] == 1 & y_bin[2:(n3+1)] == 1 & y_bin[3:(n3+2)] == 1)
  
  # Four consecutive (length 4)
  n4 <- length(y_bin) - 3
  out$u4 <- mean(y_bin[1:n4] == 1 & y_bin[2:(n4+1)] == 1 & y_bin[3:(n4+2)] == 1 & y_bin[4:(n4+3)] == 1)
  
  # Five consecutive (length 5)
  n5 <- length(y_bin) - 4
  out$u5 <- mean(y_bin[1:n5] == 1 & y_bin[2:(n5+1)] == 1 & y_bin[3:(n5+2)] == 1 & y_bin[4:(n5+3)] == 1 & y_bin[5:(n5+4)] == 1)
  
  # 10 consecutive (length 10) 
  n10 <- length(y_bin) - 9
  out$u10 <- mean(y_bin[1:n10] == 1 & y_bin[2:(n10+1)] == 1 & y_bin[3:(n10+2)] == 1 & y_bin[4:(n10+3)] == 1 & y_bin[5:(n10+4)] == 1 & y_bin[6:(n10+5)] == 1 & y_bin[7:(n10+6)] == 1 & y_bin[8:(n10+7)] == 1 & y_bin[9:(n10+8)] == 1 & y_bin[10:(n10+9)] == 1)
  
  return(out)
}

# describe_succ_unc_summ <- function(x){
#   description <- ""
#   
#   results <- successive_unchanges(x)
#   u = results$u*100
#   uu = results$uu*100
#   uuu = results$uuu*100
#   umean = u + uu * 2 + uuu * 3
#   
#   if(umean < 10){
#     description <- "Constantly changing."
#   }
#   if(umean >= 10 & umean < 25){
#     description <- "Low amount of consecutive unchanging values."
#   }
#   if(umean >= 25 & umean < 50){
#     description <- "Moderate amount of consecutive unchanging values."
#   } 
#   if(umean >= 50 & umean < 80){
#     description <- "High amount of consecutive unchanging values."
#   }
#   if(umean >= 80){
#     description <- "Very high amount of consecutive unchanging values."
#   }
#   
#   return(description)
# }


# alternatives for "consecutive unchanges"
"Successive plateaus"
"Sequential constant values"
"Repeated values"
"Consecutive stable points"
"Contiguous identical values"
"Adjacent equal values"
"Consecutive stationary points"
"Sequential unchanging values"
"Runs of constant values"
"Value persistence"

describe_succ_unc_summ <- function(x){
  description <- ""
  
  results <- successive_unchanges(x)
  u = results$u*100
  uu = results$uu*100
  uuu = results$uuu*100
  u4 = results$u4*100
  u5 = results$u5*100
  u10 = results$u10*100

  umean = uuu * 3 + u4 * 4 + u5 * 5
  if(umean < 2){
    description <- "Few flat lines."
  }
  if(umean >= 2 & umean < 30){
    description <- "Moderate flat lines."
  }
  if(umean >= 30){# 50
    description <- "Many flat lines."
  }
  
  return(description)
}


# succ <- apply(ts_sub, 1, function(x) successive_unchanges(x))
# uuu_list <- lapply(succ, function(x) x$uuu)
# u4_list <- lapply(succ, function(x) x$u4)
# u5_list <- lapply(succ, function(x) x$u5)
# umean <- unlist(uuu_list)*100 * 3 + unlist(u4_list)*100 * 4 + unlist(u5_list)*100 * 5
# hist(umean)
# abline(v = quantile(umean,c(0.1,0.25,0.5,0.75,0.9)), col = "red")
# round(quantile(umean, c(0.1,0.2,0.5,0.75,0.9)))

