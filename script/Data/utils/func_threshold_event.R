threshold_event <- function(x, th, direction=c(">", "<")[2]){
  # x is a time series vector, 
  # Return the start and end indices where x < th
  if(direction==">") x <- -x
  if(direction==">") th <- -th
  events <- list()
  if(any(x < th)){
    tag01 <- as.numeric(x < th)
    event_starts <- which(diff(c(0,tag01))>0)
    event_ends <- which(diff(c(tag01,0))<0)
    
    for(i in 1:length(event_starts)){
      events[[i]] <- c("start" = event_starts[i], 
                       "end" = event_ends[i],
                       "duration" = event_ends[i] - event_starts[i]+1)
    }
  }
  # Return the list of events
  return(events)
}
  