rm(list = ls())
library(tidyverse)
library(ggplot2)
library(ggpubr)

# ─────────────────────────────────────────────
# 1. set wd to script folder
# ─────────────────────────────────────────────
try({
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    this_file <- rstudioapi::getActiveDocumentContext()$path
  } else {
    this_file <- normalizePath(sys.frames()[[1]]$ofile)
  }
  setwd(dirname(this_file))
}, silent = TRUE)

# ─────────────────────────────────────────────
# 2. load data & keep medians
# ─────────────────────────────────────────────
df <- read_csv("../results/paper/w_df.csv") %>% filter(quantile == 50)# , w>0.5
df$dataset <- factor(
  df$dataset,
  levels = c("Synthetic w/ ground truth", "Synthetic", "Air Quality", "NICU Heart Rate")
)
colnames(df)[colnames(df) == "DTW distance decrease ↓"] <- "∆ DTW ↓"
df$`RaTS ↑`[df$`RaTS ↑` < 0] <- 0

# ─────────────────────────────────────────────
# 3. helper plot
# ─────────────────────────────────────────────
build_plot <- function(sub_df, metric_name = "RaTS ↑") {
  dot_models  <- c("InstructTime", "InstructTime (open-vocab)")
  line_models <- c("TEdit", "Time Weaver")
  
  col_map <- c(
    "InstructTime"              = "#F6A97E" ,
    "InstructTime (open-vocab)" = "#D16A87",
    "TEdit"                     = "#1F77B4",
    "Time Weaver"               = "#17A689"
  )
  
  sub_df <- sub_df %>% mutate(
    w_jit = case_when(
      Model == dot_models[1] ~ w - 0.00,
      Model == dot_models[2] ~ w + 0.00,
      TRUE                   ~ w
    )
  )
  strip_text_theme <- if (metric_name == "∆ DTW ↓") {
    element_blank()
  } else {
    element_text(face = "bold", size = 7)
  }
  
  ggplot() +
    geom_line(
      data = sub_df %>% filter(Model %in% line_models),
      aes(w, .data[[metric_name]], colour = Model),
      size = 0.5, linetype=2
    ) +
    geom_line(
      data = sub_df %>% filter(Model %in% dot_models),
      aes(w, .data[[metric_name]], colour = Model, group = Model),
      size = 0.5
    ) +
    geom_point(
      data = sub_df %>% filter(Model %in% dot_models),
      aes(w_jit, .data[[metric_name]], colour = Model),
      size = 1.
    ) +
    facet_wrap(~ dataset, nrow = 1, scales = "free_y") + # if metric is "∆ DTW ↓", no strip.text
    scale_x_continuous(breaks = sort(unique(sub_df$w))) +
    scale_y_continuous(labels = scales::label_number(accuracy = 0.01, trim = FALSE))+
    scale_color_manual(values = col_map) +
    theme_minimal(base_size = 8) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", size = 0.3),   # ← show x and y axis lines
      panel.spacing      = unit(1, "lines"),
      strip.text         = strip_text_theme,
      strip.background = element_blank(),
      legend.title       = element_blank(),
      legend.text = element_text(size = 8),
      legend.position    = "top",
      axis.title.y       = element_text(margin = margin(r = 4), size = 8)  # , face = "bold"← smaller y-label
    ) +
    labs(x = NULL, y = metric_name)
}

# ─────────────────────────────────────────────
# 4. create plots
# ─────────────────────────────────────────────
text_plot1 <- build_plot(df %>% filter(setting == "Text-based"),
                         metric_name = "RaTS ↑")
text_plot2 <- build_plot(df %>% filter(setting == "Text-based"),
                         metric_name = "∆ DTW ↓")
attr_plot1 <- build_plot(df %>% filter(setting == "Attribute-based"),
                         metric_name = "RaTS ↑")
attr_plot2 <- build_plot(df %>% filter(setting == "Attribute-based"),
                         metric_name = "∆ DTW ↓")

# ─────────────────────────────────────────────
# 5. assemble
# ─────────────────────────────────────────────
text_block <- annotate_figure( ggarrange( text_plot1, text_plot2,  ncol = 1, heights=c(1, 0.8),common.legend = TRUE, legend = "top"),
                               left = text_grob("Text-based", face = "bold", size = 9, rot = 90),
                               bottom = text_grob("Editing Strength", size = 8)) #, face = "bold"

attr_block <- annotate_figure( ggarrange( attr_plot1, attr_plot2,  ncol = 1, heights=c(1, 0.8), common.legend = TRUE, legend = "none"),
                               left = text_grob("Attribute-based", face = "bold", size = 9, rot = 90),
                               bottom = text_grob("Editing Strength", size = 8))
empty_space <- ggplot() + theme_void()
combined <- ggarrange(text_block, empty_space, attr_block, 
                      ncol = 1, 
                      heights = c(1, 0.05, 0.85), common.legend = TRUE, legend = "top")


ggsave("../results/paper/interpolated_editing.pdf", plot = combined, width = 7.5, height = 5.2, units = "in", device = cairo_pdf)

