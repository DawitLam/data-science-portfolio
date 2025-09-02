# Medical Dataset Documentation

## Overview
This documentation describes the synthetic medical datasets used across all portfolio projects. All data is synthetically generated and contains no real patient information.

## Dataset Generation Methodology

### Statistical Foundations
- Age distributions based on NHANES data for osteoporosis populations
- BMI distributions follow CDC adult population statistics
- Bone density measurements align with WHO diagnostic criteria
- Fracture rates match epidemiological studies from medical literature

### Data Quality Assurance
- Realistic statistical correlations between variables
- Proper handling of missing data patterns
- Medically plausible value ranges
- Consistent patient identifiers across datasets

---

## Dataset Schemas

### master_patient_data.csv
**Primary patient demographics and clinical measurements**

| Column | Type | Description | Range/Values | Notes |
|--------|------|-------------|--------------|-------|
| `patient_id` | String | Unique patient identifier | PT000001-PT010000 | Primary key |
| `age` | Integer | Patient age in years | 18-95 | Normal distribution, μ=68, σ=12 |
| `gender` | String | Patient gender | Female, Male | 75% female (osteoporosis population) |
| `bmi` | Float | Body Mass Index | 16-50 | Gender-specific normal distributions |
| `ethnicity` | String | Self-reported ethnicity | Caucasian, African American, Hispanic, Asian, Other | US population approximation |
| `smoking_status` | String | Current smoking status | Never, Former, Current | 45%/35%/20% distribution |
| `alcohol_units_per_week` | Integer | Weekly alcohol consumption | 0-30 | Poisson distribution, gender-specific |
| `education_level` | String | Highest education completed | High School, Some College, Bachelor's, Graduate | |
| `insurance_type` | String | Primary insurance | Medicare, Private, Medicaid, None | Age-appropriate distribution |
| `registration_date` | Date | Study enrollment date | 2018-01-01 to 2024-12-31 | YYYY-MM-DD format |
| `measurement_date` | Date | Clinical measurement date | 2023-01-01 to 2024-12-31 | Recent measurements |
| `spine_t_score` | Float | Lumbar spine bone density T-score | -4.0 to 2.0 | WHO diagnostic criteria |
| `hip_t_score` | Float | Hip bone density T-score | -4.0 to 2.0 | WHO diagnostic criteria |
| `height_cm` | Float | Height in centimeters | 140-200 | Gender-specific distributions |
| `weight_kg` | Float | Weight in kilograms | 40-150 | Calculated from BMI and height |
| `vitamin_d_ng_ml` | Float | Vitamin D level | 5-100 | Normal range: 30-100 ng/mL |
| `calcium_mg_dl` | Float | Serum calcium level | 7.0-12.0 | Normal range: 8.5-10.5 mg/dL |
| `phosphorus_mg_dl` | Float | Serum phosphorus level | 1.5-6.0 | Normal range: 2.5-4.5 mg/dL |
| `pth_pg_ml` | Float | Parathyroid hormone level | 5-200 | Normal range: 10-55 pg/mL |
| `grip_strength_kg` | Float | Hand grip strength | 10-60 | Age and gender adjusted |
| `has_fracture` | Boolean | Had fracture during study | True, False | Calculated field |
| `total_fractures` | Integer | Total fracture count | 0-3 | Count of fracture events |

### fracture_events.csv
**Detailed fracture event records**

| Column | Type | Description | Range/Values | Notes |
|--------|------|-------------|--------------|-------|
| `patient_id` | String | Links to master patient data | PT000001-PT010000 | Foreign key |
| `fracture_id` | String | Unique fracture identifier | FX000001-FX999999 | Primary key |
| `fracture_date` | Date | Date of fracture occurrence | 2022-01-01 to 2024-12-31 | YYYY-MM-DD format |
| `fracture_site` | String | Anatomical location | Hip, Vertebral, Wrist, Shoulder, Ankle, Ribs, Pelvis | Hip most common (35%) |
| `severity` | String | Fracture severity | Minor, Moderate, Severe | Clinical assessment |
| `treatment` | String | Treatment approach | Conservative, Surgical fixation, Joint replacement | |
| `recovery_days` | Integer | Recovery time in days | 14-300 | Severity-dependent Poisson |
| `mechanism` | String | Cause of fracture | Fall from standing, Fall from height, MVA, Sports, Pathological, Unknown | Most are low-energy falls |

### patient_surveys.csv
**Patient-reported outcome measures**

| Column | Type | Description | Range/Values | Notes |
|--------|------|-------------|--------------|-------|
| `patient_id` | String | Links to master patient data | PT000001-PT010000 | Foreign key |
| `survey_date` | Date | Survey completion date | 2023-01-01 to 2024-12-31 | YYYY-MM-DD format |
| `quality_of_life_physical` | Integer | Physical QOL score | 0-100 | Higher scores = better QOL |
| `quality_of_life_mental` | Integer | Mental QOL score | 0-100 | Higher scores = better QOL |
| `pain_level_0_10` | Integer | Current pain level | 0-10 | 0=no pain, 10=severe pain |
| `mobility_status` | String | Current mobility | Independent, Uses walking aid, Assisted walking, Wheelchair | Functional assessment |
| `falls_past_year` | Integer | Number of falls in past year | 0-10 | Risk factor for future fractures |
| `fear_of_falling_1_5` | Integer | Fear of falling scale | 1-5 | 1=not afraid, 5=very afraid |
| `care_satisfaction_1_5` | Integer | Satisfaction with care | 1-5 | 1=very dissatisfied, 5=very satisfied |
| `dietary_calcium_adequate` | Boolean | Adequate dietary calcium | True, False | Self-reported dietary assessment |
| `exercise_minutes_per_week` | Integer | Weekly exercise duration | 0-500 | Self-reported physical activity |

### clinical_notes.txt / clinical_notes.json
**Unstructured clinical documentation**

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| `patient_id` | String | Links to master patient data | Foreign key |
| `note_id` | String | Unique note identifier | NOTE000001-NOTE999999 |
| `note_date` | Date | Date note was written | YYYY-MM-DD format |
| `note_type` | String | Type of clinical note | Progress Note, Consultation, Follow-up |
| `provider` | String | Clinician who wrote note | Dr. [Surname] |
| `note_text` | String | Full clinical note content | Structured medical note format |

---

## Clinical Context

### Fracture Risk Factors
**Primary Risk Factors** (well-established):
- Age (especially >65 years)
- Gender (female > male)
- Low bone mineral density (T-score < -2.5)
- Previous fracture history
- Family history of fractures

**Secondary Risk Factors** (modifiable):
- Low BMI (<20 kg/m²)
- Smoking (current or former)
- Excessive alcohol consumption
- Physical inactivity
- Vitamin D deficiency
- Certain medications

### Clinical Measurements

**Bone Density (T-scores)**
- **Normal**: T-score ≥ -1.0
- **Osteopenia**: T-score -1.0 to -2.5
- **Osteoporosis**: T-score < -2.5

**Laboratory Reference Ranges**
- **Vitamin D**: 30-100 ng/mL (optimal)
- **Calcium**: 8.5-10.5 mg/dL
- **Phosphorus**: 2.5-4.5 mg/dL
- **PTH**: 10-55 pg/mL

---

## Data Ethics and Privacy

### Synthetic Data Assurance
- **No real patient data**: All data is computationally generated
- **Realistic distributions**: Based on published medical literature
- **Privacy by design**: No identifiable information patterns
- **Medical validity**: Clinically plausible relationships and values

### Usage Guidelines
- Data is intended for portfolio demonstration only
- Should not be used for actual clinical decision-making
- Appropriate for educational and research purposes
- Maintains patient privacy standards

---

## Data Validation

### Automated Quality Checks
- **Range validation**: All values within medically plausible ranges
- **Relationship validation**: Logical relationships between related variables
- **Distribution validation**: Statistical distributions match expected patterns
- **Completeness validation**: Missing data patterns are realistic

### Known Limitations
- Simplified compared to real electronic health records
- Limited longitudinal data (2-3 year window)
- Focused on fracture risk domain
- Does not include imaging data or complex temporal patterns

---

## Future Enhancements

### Potential Dataset Expansions
- Longitudinal measurements over longer time periods
- Integration of imaging data (DICOM metadata)
- Medication adherence tracking
- Healthcare utilization patterns
- Social determinants of health data

### Technical Improvements
- Real-time data generation for streaming applications
- Integration with synthetic EHR systems
- Advanced temporal modeling for disease progression
- Multi-site data generation for federated learning scenarios

---

## References and Sources

Data generation methodology references:
1. WHO Fracture Risk Assessment Tool (FRAX)
2. National Osteoporosis Foundation Clinical Guidelines
3. NHANES Bone Health Survey Data
4. International Osteoporosis Foundation Statistics
5. Medicare Claims Analysis for Fracture Epidemiology

*For specific medical literature references, see individual project documentation.*
