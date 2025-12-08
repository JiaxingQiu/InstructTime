# To find the onset time of the start time point where the series begins to drop towards a threshold, and the offset time of the end time points where the series begins to rise back to or above the threshold after having dropped below it
threshold_event_extra <- function(x,  th, direction=c(">", "<")[2], min_dur = 2, plot=F) {
  
  event_pairs <- threshold_event(x, th, direction)
  event_pairs_extra <- list()
  if(length(event_pairs)>0){
    # prepare a set of smoothed x, to calculate the estimate of onset/offset
    x_smoothed <- list()
    time_index <- seq_along(x)
    for(k in c(5, 10, 15, 20) ){
      x_sm <- lowess(time_index, x, f = k/300)
      x_smoothed[[paste0("k=",k)]] <- x_sm$y
    }
    for(i in c(1:length(event_pairs))){
      event_start = event_pairs[[i]][1]
      event_end = event_pairs[[i]][2]
      # for each smoothed x
      event_onset <- c()
      event_offset <- c()
      for(j in c(1:length(x_smoothed))){
        x_smooth <- x_smoothed[[j]]
        x_smooth_diff <- c(0,diff(x_smooth))
        # find the onset timepoint before event_start: the last below zero point in x_smooth_diff before event_start
        idx <- which(x_smooth_diff>=0) 
        event_onset <- c(event_onset, max(idx[which(idx < event_start)]))
        
        # find the offset timepoint after event_end
        idx <- which(x_smooth_diff<=0) 
        event_offset <- c(event_offset, min(min(idx[which(idx > event_end)])-1, length(x)) )
      }
      event_onset_est <- round(median(event_onset))
      event_offset_est <- round(median(event_offset))
      event_pairs_extra[[i]] <- c(event_start,
                                  event_end, 
                                  "onset" = event_onset_est, 
                                  "offset" = event_offset_est)
      if(plot){
        # make a plot of x and highlight the four time points
        plot(time_index, x, type = "l", col = "red", main = "Event timepoints", xlab = "Time", ylab = "Value", ylim = range(x))
        for(x_smooth in x_smoothed){
          lines(time_index, x_smooth, col = "blue", lwd = 0.5)
        }
        # add points at four x locations c(event_start, event_end, event_onset_est, event_offset_est) 
        points(event_start, x[event_start], col = "red", pch = 19, cex = 1)
        points(event_end, x[event_end], col = "red", pch = 19, cex = 1)
        points(event_onset_est, x[event_onset_est], col = "orange", pch = 19, cex = 1)
        points(event_offset_est, x[event_offset_est], col = "orange", pch = 19, cex = 1)
      }
    }
  }
  return(event_pairs_extra)
}

