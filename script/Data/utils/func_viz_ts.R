
view_k_row <- function(ts, k_row=10, i_row=NULL, vs = c("HR","SP", "XC")[1]){
  ts_sub <- ts[1:k_row, 3:ncol(ts)]
  if(!is.null(i_row)){
    ts_sub <- ts[i_row, 3:ncol(ts)]
  }
  ts_sub$row <- seq_len(nrow(ts_sub))
  ts_sub_melted <- melt(ts_sub, id.vars = "row")
  p <- ggplot(ts_sub_melted, aes(x = variable, y = value, group = row, color = as.factor(row))) +
    geom_line() 
  if(vs == "HR") {
    p <- ggplot(ts_sub_melted, aes(x = variable, y = value, group = row)) +
      geom_line(color = "#F8766D") +
      geom_hline(aes(yintercept=80),size=0.5,linetype="dashed", color="darkgray") +
      geom_hline(aes(yintercept=90),size=0.5,linetype="dashed", color="gray")
    
  }
  if(vs == "SP") {
    p <- ggplot(ts_sub_melted, aes(x = variable, y = value, group = row)) +
      geom_line(color = "#619CFF") +
      geom_hline(aes(yintercept=80),size=0.5,linetype="dashed", color="darkgray") +
      geom_hline(aes(yintercept=90),size=0.5,linetype="dashed", color="gray")
    
  }
  if(vs == "XC"){
    xc = round(as.numeric(cor(unlist(ts_sub[1,]),unlist(ts_sub[2,]))),2)
    p <- p + labs(x = "Time (2s)", color = NULL) +
      theme_classic() +
      scale_color_manual(values = c("#F8766D", "#619CFF"))+
      labs(subtitle = paste0("cross-correlation = ",xc))+
      theme(legend.position = "none",
            axis.text.x = element_blank())
  }
  
  return(p)
}

