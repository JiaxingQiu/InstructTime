successive_increases <- function(y) {
  # Input: y - numeric vector of heart rate measurements
  # Output: list containing counts and proportions of successive increases
  
  # Convert to binary: 1 for increase, 0 for decrease or same
  N <- length(y)
  if (N < 5) {
    warning("Time series too short")
    return(NULL)
  }
  
  # Get binary sequence (1 for increase, 0 for decrease/same)
  y_diff <- diff(y)  # Get differences
  y_bin <- as.numeric(y_diff > 0)  # 1 for increase, 0 otherwise
  
  # Initialize output list
  out <- list()
  
  # Single increases (length 1)
  out$u <- mean(y_bin == 1)  # proportion of increases
  out$d <- mean(y_bin == 0)  # proportion of decreases/same
  
  # Two consecutive (length 2)
  n2 <- length(y_bin) - 1
  out$uu <- mean(y_bin[1:n2] == 1 & y_bin[2:(n2+1)] == 1)  # up,up
  out$ud <- mean(y_bin[1:n2] == 1 & y_bin[2:(n2+1)] == 0)  # up,down
  out$du <- mean(y_bin[1:n2] == 0 & y_bin[2:(n2+1)] == 1)  # down,up
  out$dd <- mean(y_bin[1:n2] == 0 & y_bin[2:(n2+1)] == 0)  # down,down
  
  # consec_ones <- function(y_bin, k) {
  #   n <- length(y_bin) - k + 1
  #   if (n <= 0) return(0)
  #   mean(Reduce(`&`, lapply(0:(k-1), function(i) y_bin[(1 + i):(n + i)]) ) == 1)
  # }
  # out$uuu <- consec_ones(y_bin, 3)
  # out$u4 <- consec_ones(y_bin, 4)
  # out$u5 <- consec_ones(y_bin, 5)

  # Three consecutive (length 3)
  n3 <- length(y_bin) - 2
  out$uuu <- mean(y_bin[1:n3] == 1 & y_bin[2:(n3+1)] == 1 & y_bin[3:(n3+2)] == 1)
  n4 <- length(y_bin) - 3
  out$u4 <- mean(y_bin[1:n4] == 1 & y_bin[2:(n4+1)] == 1 & y_bin[3:(n4+2)] == 1 & y_bin[4:(n4+3)] == 1)
  n5 <- length(y_bin) - 4
  out$u5 <- mean(y_bin[1:n5] == 1 & y_bin[2:(n5+1)] == 1 & y_bin[3:(n5+2)] == 1 & y_bin[4:(n5+3)] == 1 & y_bin[5:(n5+4)] == 1)
  
  # Calculate entropy for each length
  entropy <- function(p) {
    p <- p[p > 0]  # Remove zeros
    -sum(p * log(p))
  }
  
  out$h <- entropy(c(out$u, out$d))  # entropy of length 1
  out$h2 <- entropy(c(out$uu, out$ud, out$du, out$dd))  # entropy of length 2
  
  return(out)
}

# Example usage:
# Assuming x is your 10-minute heart rate data sampled every 2 seconds
# x should be a numeric vector of length 300 (10 min * 30 samples/min)
# 
# # Example data
# x <- rnorm(300)  # Replace with your actual heart rate data
# results <- successive_increases(x)
# 
# # Print results
# print("Proportion of increases:")
# print(results$u)
# print("Proportion of two consecutive increases:")
# print(results$uu)
# print("Entropy of patterns:")
# print(results$h)



# describing the level of consecutive increases
describe_succ_inc <- function(x){
  description <- "There is no consecutive increase."
  description2 <- ""
  description3 <- ""
  
  results <- successive_increases(x)
  if(round(results$uu*100)>1){
    description2 <- paste0("There are ", round(results$uu*100), " percent two consecutive increases;")
  }
  if(round(results$uuu*100)>1){
    description3 <- paste0(round(results$uuu*100), " percent three consecutive increases.")
  }
  
  if(any(description2!="", description3!="")){
    description <- paste0(description2, " ", description3)
  }
  
  return(description)
}


describe_succ_inc_summ <- function(x){
  description <- NA
  
  results <- successive_increases(x)
  uu = results$uu*100
  uuu = results$uuu*100
  # umean = uuu 
  u4 = results$u4*100
  u5 = results$u5*100
  umean = uuu*3 + u4*4 + u5*5
  
  if(umean < 105){ # 5, raw data version
    description <- "Low amount of consecutive increases."
  }
  if(umean >= 150 & umean < 170){ # 5-15
    description <- "Moderate amount of consecutive increases."
  } 
  if(umean >= 215){ # 15
    description <- "High amount of consecutive increases."
  }
  
  return(description)
}


# succ <- apply(ts_sub, 1, function(x) successive_increases(x))
# u3 <- lapply(succ, function(x) x$uuu)
# u4 <- lapply(succ, function(x) x$u4)
# u5 <- lapply(succ, function(x) x$u5)
# umean <- unlist(u3)*100*3 + unlist(u4)*100*4 + unlist(u5)*100*5
# hist(umean)
# abline(v = quantile(umean, c(0.15,0.4,0.5,0.6,0.85)), col = "red")
# round(quantile(umean, c(0.15,0.4,0.5,0.6,0.85)))

# uu_list <- lapply(succ, function(x) x$uu)
# uuu_list <- lapply(succ, function(x) x$uuu)
# hist(unlist(uu_list)*100, breaks = 50)
# abline(v = quantile(unlist(uu_list)*100, c(0.025,0.25,0.5,0.75,0.975)), col = "red")
# round(quantile(unlist(uu_list)*100, c(0.025,0.25,0.5,0.75,0.975)))
# hist(unlist(uuu_list)*100)
# abline(v = quantile(unlist(uuu_list)*100, c(0.1,0.25,0.5,0.75,0.9)), col = "red")
# round(quantile(unlist(uuu_list)*100, c(0.1,0.25,0.5,0.75,0.9)))
