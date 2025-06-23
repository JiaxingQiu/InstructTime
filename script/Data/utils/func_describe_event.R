
describe_brady_event <- function(x, th, direction = "<", plot = F, type = c(seq(1,4),0)[1]){
  
  if(type == 0){
    type <- sample(seq(1,4),1)
  }
  # description string to return
  description <- ""
  context <- paste0("This is a heart rate time series from a NICU infant. A Bradycardia ",th," event is defined as heart rate below ",th," beats per minute. \n ")
  events <- threshold_event_filtered(x, th, direction, plot=plot)
  event_descriptions <- list()
  if(length(events)>0){
    for(i in seq_along(events)){
      event <- events[[i]]
      # ground-truth description generation
      # easy
      if(type == 1){
        event_descriptions[[i]] <- paste0("There was a Bradycardia ",th," event from time ", event['start'], " to time ",event['end'],".")
      }
      if(type == 2){
        event_descriptions[[i]] <- paste0("Since time ",round(event['start'])," the heart rate remained below ",th," for ",event['duration']," time points. ",
                                          "Then the heart rate increased to around ",x[event['offset']], " beats per minute",".")
      }
      # hard
      if(type == 3) {
        event_descriptions[[i]] <- paste0("Since time ", round(event['onset']), ", ",
                                          "the heart rate began to drop, falling below ",th, " beats per minute"," at time ", round(event['start']), ", ",
                                          "and remained below ",th," for the next ", round(event['duration']), " time points.")
      }
      if(type == 4){
        drop_fast <- event['drop_rate'] < (-5)
        back_fast <- event['drop_rate'] > 5
        event_descriptions[[i]] <- paste0("The heart rate started to drop from about ", x[event['onset']], " beats per minute", ifelse(drop_fast, " fastly ", " "), "since time ", round(event['onset']),". ",
                                          "After remaining below ",th, " beats per minute"," for ",event['duration']," time points, ",
                                          "the heart rate increased to around ", x[event['offset']], " beats per minute", ifelse(back_fast, " fastly.",".") )
      }
      event_descriptions[[i]]  <- paste0("Event ",i,": ", event_descriptions[[i]])
    }
  }
  
  if(length(event_descriptions)>0){
    description <- paste0(context, paste0(event_descriptions, collapse = " \n "))
  }
  
  return(description)
}




get_most_severe_event <- function(row_index){
  # for each row of ts_sub
  # find the most severe event in events80, events90 and events100
  # keep the most severe event
  # if no event, return "no event." 
  event80 <- events80[row_index]
  event90 <- events90[row_index]
  event100 <- events100[row_index]
  
  if(all(c(event80, event90, event100) == "")){
    return(NA)
  }
  if(event80!=""){
    return(event80)
  }
  if(event90!=""){
    return(event90)
  }
  if(event100!=""){
    return(event100)
  }
}




# # --- test ---
# th <- 80
# rows <- apply(ts_hr[,3:ncol(ts_hr)], 1, function(x) any(x < th))
# ts_sub <- ts_hr[rows,3:ncol(ts_hr)]
# x = as.numeric(unlist(ts_sub[2,]))
# th = 100
# direction = "<"
# # --- evaluate population dropping speed ---
# duration_all <- c()
# drop_rate_all <- c()
# back_rate_all <- c()
# for(i in 1:nrow(ts_sub)){
#   x <- as.numeric(unlist(ts_sub[i,]))
#   events <- threshold_event_filtered(x, th = 100, direction = "<", plot=F)
#   for(j in seq_along(events)){
#     duration_all <- c(duration_all, events[[j]]['duration'])
#     drop_rate_all <- c(drop_rate_all, events[[j]]['drop_rate'])
#     back_rate_all <- c(back_rate_all, events[[j]]['back_rate'])
#   }
# }
# 
# hist(duration_all, breaks=100)
# percentiles <- quantile(duration_all, c(0.25, 0.75))
# abline(v = percentiles, col = "red", lwd = 2)
# hist(drop_rate_all, breaks=100)
# percentiles <- quantile(drop_rate_all, c(0.25, 0.75))
# abline(v = percentiles, col = "red", lwd = 2)
# hist(back_rate_all, breaks=100)
# percentiles <- quantile(back_rate_all, c(0.25, 0.75))
# abline(v = percentiles, col = "red", lwd = 2)

