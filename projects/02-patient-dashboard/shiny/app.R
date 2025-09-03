library(shiny)
library(ggplot2)
library(dplyr)
library(DT)

data_dir <- normalizePath(file.path('..', '..', 'data', 'synthetic'), mustWork = FALSE)
cvd_path <- file.path(data_dir, 'cardiovascular_risk_data.csv')
master_path <- file.path(data_dir, 'master_patient_data.csv')
fracture_path <- file.path(data_dir, 'fracture_events.csv')

load_safe <- function(path){
  if(file.exists(path)){
    tryCatch(read.csv(path, stringsAsFactors = FALSE), error = function(e) NULL)
  } else NULL
}

cvd_df <- load_safe(cvd_path)
master_df <- load_safe(master_path)
fracture_df <- load_safe(fracture_path)

ui <- navbarPage("Patient Dashboard (Shiny)",
  tabPanel("Overview",
    fluidRow(
      column(4, wellPanel(h3("Datasets"),
                         p(ifelse(!is.null(master_df), paste0('Master patients: ', nrow(master_df)), 'Master dataset missing')),
                         p(ifelse(!is.null(cvd_df), paste0('Cardiovascular: ', nrow(cvd_df)), 'Cardio dataset missing')),
                         p(ifelse(!is.null(fracture_df), paste0('Fracture events: ', nrow(fracture_df)), 'Fracture dataset missing'))
      )),
      column(8, plotOutput('age_hist'))
    )
  ),
  tabPanel("Cardiovascular",
    fluidRow(
      column(6, plotOutput('systolic_dist')),
      column(6, DT::dataTableOutput('cvd_table'))
    )
  ),
  tabPanel("Fracture",
    fluidRow(
      column(6, DT::dataTableOutput('fracture_table')),
      column(6, verbatimTextOutput('fracture_note'))
    )
  ),
  tabPanel("Patient Lookup",
    sidebarLayout(
      sidebarPanel(textInput('patient_id', 'Enter patient_id (e.g. CVD_000001)'), actionButton('search', 'Search')),
      mainPanel(DT::dataTableOutput('patient_result'))
    )
  )
)

server <- function(input, output, session){
  output$age_hist <- renderPlot({
    if(is.null(master_df)) return(NULL)
    ggplot(master_df, aes(x=age)) + geom_histogram(bins=20, fill='#2c7fb8') + theme_minimal()
  })

  output$systolic_dist <- renderPlot({
    if(is.null(cvd_df)) return(NULL)
    ggplot(cvd_df, aes(x=systolic_bp)) + geom_histogram(bins=30, fill='#de2d26') + theme_minimal()
  })

  output$cvd_table <- DT::renderDataTable({
    if(is.null(cvd_df)) return(NULL)
    DT::datatable(head(cvd_df, 200))
  })

  output$fracture_table <- DT::renderDataTable({
    if(is.null(fracture_df)) return(NULL)
    DT::datatable(head(fracture_df, 200))
  })

  output$fracture_note <- renderText({
    if(is.null(fracture_df)) return('Fracture dataset not found. Generate synthetic data via Python generator: see README.')
    sprintf('Unique fracture types: %s', paste(unique(fracture_df$fracture_type), collapse=', '))
  })

  observeEvent(input$search, {
    pid <- input$patient_id
    res <- NULL
    tables <- list(master=master_df, cvd=cvd_df, fracture=fracture_df)
    for(nm in names(tables)){
      df <- tables[[nm]]
      if(!is.null(df) && 'patient_id' %in% names(df)){
        sel <- df[df$patient_id == pid, , drop=FALSE]
        if(nrow(sel) > 0){
          res <- sel
          break
        }
      }
    }
    if(is.null(res)) res <- data.frame(message = 'Patient not found in loaded datasets')
    output$patient_result <- DT::renderDataTable(DT::datatable(res))
  })
}

shinyApp(ui, server)
