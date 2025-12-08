# # describing heart rate variability
# describe_hr_histogram <- function(x){
#   
#   description <- ""
#   description_constant <- ""
#   description_normal <- ""
#   description_high_var <- ""
# 
# 
#   median_hr <- round(median(x), 2)
#   if(all(x >= median_hr - 10 & x <= median_hr + 10)){
#     description_constant <- "Low variability. "
#   }
#   if(all(x >= median_hr - 5 & x <= median_hr + 5)){
#     description_constant <- "Very low variability. "
#   }
#   
# 
# 
#   if(all(x >= 120 & x <= 140)){
#     description_normal <- "It is within a normal range. "
#   }
# 
#   if(any(x > median_hr + 10 | x < median_hr - 10) ){
#     description_high_var <- "Moderate variability. "
#   }
#   if(any(x > median_hr + 20 | x < median_hr - 20) ){
#     # run outlier detection on x using boxplot.stats
#     boxplot_stats <- boxplot.stats(x)
#     outliers <- boxplot_stats$out
#     if(length(outliers) >= 0.05*length(x)){ # more than 5% of the data are outliers
#       description_high_var <- "High variability. "
#     }
#   }
#   
#   if(description_constant!=""){
#     description <- paste0(description, description_constant)
#   }
#   if(description_normal!=""){
#     description <- paste0(description, description_normal)
#   }
#   if(description_high_var!=""){
#     description <- paste0(description, description_high_var)
#   }
# 
#   
#   return(description)
# }
# # plot a histogram of all cells in dataframe ts_sub
# hist(as.matrix(ts_sub), breaks = 100)
# # add vertical lines of quantile 25, 50, 75
# abline(v = quantile(as.matrix(ts_sub), 0.25), col = "red")
# abline(v = quantile(as.matrix(ts_sub), 0.5), col = "green")
# abline(v = quantile(as.matrix(ts_sub), 0.75), col = "blue")
# quantile(as.matrix(ts_sub), c(0.05, 0.5, 0.95))




# ts_sub is a dataframe with 1:300 columns each row is a time series of length 300
# for each row calculate the mean, median, and standard deviation, and the 5th, 25th, 50th, 75th, and 95th percentiles
get_stats <- function(x){
  mean <- mean(x)
  std <- sd(x)
  quantiles <- round(as.numeric(quantile(x, c(0.05, 0.25, 0.5, 0.75, 0.95))))
  # Compute local variability: mean std over every 10 timepoints (non-overlapping)
  n <- length(x)
  block_size <- 10
  n_blocks <- floor(n / block_size)
  block_stds <- sapply(1:n_blocks, function(i) {
    start <- (i - 1) * block_size + 1
    end <- i * block_size
    sd(x[start:end])
  })
  local_variability <- median(block_stds)

  stats <- list(mean = mean, 
       std = local_variability,#std,
       q5 = quantiles[1],
       q25 = quantiles[2],
       q50 = quantiles[3],
       q75 = quantiles[4],
       q95 = quantiles[5])
  return(stats)
}
# # row-wise apply get_stats to ts_sub
# stats <- apply(ts_sub, 1, get_stats)
# # convert stats to dataframe properly
# stats_df <- do.call(rbind.data.frame, stats) # or alternatively stats_df <- as.data.frame(t(sapply(stats, unlist)))
# st <- stats_df$std
# hist(st, breaks = 100)
# abline(v = quantile(st, 0.05), col = "red")
# abline(v = quantile(st, 0.25), col = "red")
# abline(v = quantile(st, 0.5), col = "green")
# abline(v = quantile(st, 0.75), col = "blue")
# abline(v = quantile(st, 0.95), col = "blue")
# quantile(st, c(0.05, 0.25, 0.5, 0.75, 0.95))

# describing heart rate variability
describe_hr_histogram <- function(x){
  
  description <- ""
  description_normal <- "" # normal range (mean)
  description_var <- "" # variance
  description_out <- "" # outliers
  
  stats <- get_stats(x)
  
  
  # if(all(x >= 120 & x <= 140)){
  #   description_normal <- "It is within the normal range."
  # }
  # if(all(x < 120)){
  #   description_normal <- "Lower than the normal range."
  # }
  # if(all(x > 140)){
  #   description_normal <- "Higher than the normal range."
  # }
  
  
  if(stats$std<1.5){
    description_var <- "Low variability."
  }
  if(stats$std>=1.5 & stats$std<=3){
    description_var <- "Moderate variability."
  }
  if(stats$std>3){
    description_var <- "High variability."
  }
  
  median_hr <- stats$q50
  if(any(x > median_hr + 20 | x < median_hr - 20)){
    boxplot_stats <- boxplot.stats(x)# run outlier detection on x using boxplot.stats
    outliers <- boxplot_stats$out
    if(length(outliers) < 0.25*length(x)){ # less than ~0% of the data are outliers
      description_out <- "A few outliers."
    }
    # if(length(outliers) >= 0.25*length(x)){ # more than ~0% of the data are outliers
    #   description_out <- "Many outliers."
    # }
  }
  # if(description_normal!=""){
  #   description <- paste0(description,
  #                         ifelse(description == "", "", " "),
  #                         description_normal)
  # }
  if(description_var!=""){
    description <- paste0(description, 
                          ifelse(description == "", "", " "),
                          description_var)
  }
  # if(description_out!=""){
  #   description <- paste0(description,
  #                         ifelse(description == "", "", " "),
  #                         description_out)
  # }
  
  
  return(description)
}
