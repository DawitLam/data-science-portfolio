Patient Dashboard (Streamlit)

This is a minimal interactive dashboard that demonstrates cohort and patient-level views using the repository's synthetic medical data.

Features
- Overview: cohort counts and age distribution
- Cardiovascular tab: basic distributions and 10-year high-risk prevalence
- Fracture tab: basic event counts
- Patient lookup: search by `patient_id`

How it works
- The app attempts to load datasets from `data/synthetic/`:
  - `cardiovascular_risk_data.csv`
  - `master_patient_data.csv`
  - `fracture_events.csv`
- If files are missing, the app will try to call the shared generators in `shared/data_generators/` to create small synthetic datasets.

Run locally (PowerShell)
```powershell
cd projects\02-patient-dashboard
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

## R / Shiny instructions
If you prefer to run an R Shiny app (useful for interview demonstrations), a minimal Shiny scaffold is included at `projects/02-patient-dashboard/shiny/app.R`.

Required R packages:
- shiny
- ggplot2
- dplyr
- DT

Run the Shiny app (from project root PowerShell):
```powershell
# start an R session or run this from RScript if available
R -e "shiny::runApp('projects/02-patient-dashboard/shiny', launch.browser=TRUE)"
```

Notes:
- The Shiny app reads the same `data/synthetic/` files as the Streamlit app.
- Keeping both Python (Streamlit) and R (Shiny) versions lets you highlight both stacks during interviews.
