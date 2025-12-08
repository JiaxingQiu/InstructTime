threshold_event_filtered <- function(x, th, direction = "<", plot = F){
  
  events <- threshold_event_extra(x, th, direction)
  events <- lapply(events, function(event) {
    return(c(event, 
             "duration" = as.numeric(event['end'] - event['start'] + 1),
             "drop_rate" = as.numeric(x[event['start']] - x[event['onset']]) / as.numeric(event['start'] - event['onset'] + 1),
             "back_rate" = as.numeric(x[event['offset']] - x[event['end']]) / as.numeric(event['offset'] - event['end'] + 1)
    )) })
  
  if(length(events)>0){
    # filter 1: duration in [5, 150) => [10, 300)s # filter 2: drop_rate < 0 and back_rate > 0 
    indices <- sapply(events, function(event) event['duration'] >= 1 & event['duration'] < 150 & event['drop_rate'] < 0 & event['back_rate'] > 0 )
    events <- events[indices]
  }
  if(length(events)>0){
    # Remove empty elements
    events <- events[sapply(events, function(x) !is.null(x) && length(x) != 0)]
  }
    
  if(length(events)>0){
    if(plot){
      time_index <- seq_along(x)
      for(event in events){
        plot(time_index, x, type = "l", col = "blue", main = "Event timepoints", xlab = "Time", ylab = "Value", ylim = range(x))
        points(event['start'], x[event['start']], col = "red", pch = 19, cex = 1)
        points(event['end'], x[event['end']], col = "red", pch = 19, cex = 1)
        points(event['onset'], x[event['onset']], col = "orange", pch = 19, cex = 1)
        points(event['offset'], x[event['offset']], col = "orange", pch = 19, cex = 1)
      }
    }
  }
  
  return(events)
  
}
