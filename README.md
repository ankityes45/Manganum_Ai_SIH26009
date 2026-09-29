# Manganum AI — SIH26009

### AI/ML and Space Technology for Manganese Reserve Intelligence and Production Shortfall Prevention

**Team: Sovereign**

Manganum AI is a decision-support prototype developed for Smart India Hackathon problem statement SIH26009:

> Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome Production Shortfalls

The system combines geological information, environmental indicators, equipment performance, blasting delays, historical production, and production targets to support mine planning and operational decisions.

## Problem Statement Alignment

| Requirement | Prototype implementation |
|---|---|
| Identify and map manganese reserves | Geospatial prospectivity model and interactive map |
| Surface indicators | Elevation, slope, NDVI, spectral and geological indices |
| Sub-surface indicators | Depth to ore, ore thickness, bulk density, Mn grade |
| Space/environment inputs | Rainfall, soil moisture, NDVI, land temperature |
| Equipment constraints | Downtime, availability, utilization, maintenance |
| Blasting constraints | Blast delay, blast efficiency, planned vs actual output |
| Production prediction | XGBoost production regression model |
| Production shortfall | Production target versus predicted production |
| Shortfall driver analysis | Equipment, blasting and environmental risk scores |
| Corrective actions | Scheduling, maintenance, blasting and equipment-redeployment actions |
| User-friendly dashboard | Streamlit, Folium and Plotly |
| Human-in-the-loop validation | Geologist field override endpoint and dashboard form |

## End-to-End Workflow

```text
Study Area
    ↓
Environmental / Space-Technology Check
    ↓
Equipment Performance Check
    ↓
Blasting Delay Check
    ↓
Production Target + Historical Production
    ↓
Feature Integration
    ↓
XGBoost Production Prediction
    ↓
Production Shortfall Calculation
    ↓
Risk / Driver Analysis
    ↓
Preventive Measures for Mine Manager

                 ╲
                  ╲
                   → Surface + Sub-surface Geological Features
                        ↓
                    XGBoost Prospectivity Model
                        ↓
                  Potential Ore / Mn Estimation
                        ↓
                    Geospatial Reserve Map

                        ↓
                    FastAPI Backend
                        ↓
                    Streamlit Dashboard
```

# Main Components

## 1. Integrated ML Notebook

**File:** `code_SIH26009_final.ipynb`

The notebook is the main data-processing and model-building pipeline.

1. Load the study area.
2. Check environmental and space-technology conditions.
3. Check equipment performance.
4. Check blasting delays.
5. Load production targets and historical production.
6. Integrate the operational datasets.
7. Train the production prediction model and calculate shortfall.
8. Analyse shortfall drivers.
9. Generate preventive recommendations.
10. Build the reserve/prospectivity model using surface and sub-surface indicators.
11. Estimate potential ore by zone.
12. Save outputs for FastAPI and Streamlit.

## 2. Geological / Reserve Intelligence

**File:** `02_finalize_prospectivity_model.py`

### Surface indicators

- `elevation_m`
- `slope_deg`
- `ndvi`
- `iron_oxide_ratio`
- `fe_mn_swir_ratio`
- `alteration_index`
- `bare_ground_index`

### Sub-surface indicators

- `depth_to_ore_m`
- `ore_thickness_m`
- `bulk_density_t_m3`
- `mn_grade_pct`

The model uses drill-hole labels and an XGBoost classifier to produce a spatial prospectivity surface. A spatial holdout is used for prototype validation using precision, recall and F1 score.

Potential ore and contained Mn are estimated from grid-level geological properties and weighted by prospectivity score.

## Environmental / Space-Technology Layer

The environmental dataset contains:

- Rainfall (`rainfall_mm`)
- Soil moisture (`soil_moisture`)
- Vegetation index (`ndvi`)
- Land temperature (`land_temperature_c`)

These variables are summarized by zone and used in environmental risk assessment.

## Equipment Intelligence

The equipment dataset contains:

- Equipment ID and type
- Hours available
- Hours operated
- Downtime hours
- Maintenance hours
- Availability
- Utilization
- Downtime reason

The backend identifies the equipment unit with the highest accumulated downtime for the selected zone.

## Blasting Intelligence

The blasting dataset contains:

- Blast ID
- Planned and actual start time
- Delay hours
- Planned ore
- Actual ore
- Blast efficiency
- Delay reason

Recent blast performance is summarized by zone and used for blasting-risk assessment.

## Production Prediction and Shortfall

The production model uses historical production, environmental indicators, equipment indicators, blasting indicators, and production targets.

The model predicts next-day production in tonnes per day.

```text
Shortfall = max(Production Target - Prediction, 0)
```

```text
Shortfall % = Shortfall / Production Target × 100
```

Prototype decision bands:

```text
< 5%       → LOW
5%–<15%    → MEDIUM
≥ 15%      → HIGH
```

## Risk Driver Analysis

The system evaluates three operational risk categories:

```text
Equipment
Blasting
Environment
```

Their scores are normalized to provide a relative risk-driver breakdown.

## Preventive Measures

The decision layer generates manager-oriented actions such as:

- Prioritizing high-prospectivity ore blocks in the near-term mine schedule
- Maintaining a production buffer during moderate shortfall risk
- Scheduling preventive maintenance for high-downtime equipment
- Assessing available backup-equipment redeployment
- Re-planning blasting windows
- Verifying drill and haulage readiness before blasting
- Adjusting mine and haulage schedules around environmental risk windows
- Maintaining contingency capacity during higher-risk periods

# FastAPI Backend

**File:** `app.py`

FastAPI acts as the backend interface between the generated model/data artifacts and the Streamlit dashboard.

### Main endpoints

| Endpoint | Purpose |
|---|---|
| `/` | API status and available zones |
| `/health` | Health and artifact check |
| `/area` | Study-area bounds and grid information |
| `/environment` | Environmental indicators |
| `/equipment` | Equipment performance |
| `/blasting` | Blasting performance |
| `/production-target/{zone}` | Production target |
| `/production-history/{zone}` | Historical production |
| `/prediction/{zone}` | Production-model prediction |
| `/shortfall/{zone}` | Shortfall and driver analysis |
| `/forecast/{zone}` | Production forecast |
| `/recommendation/{zone}` | Preventive actions |
| `/reserve-summary` | Potential ore and Mn summary |
| `/reserve-map` | Spatial prospectivity and reserve grid |
| `/drill-holes` | Drill-hole data |
| `/model-info` | Model and feature information |
| `/overrides` | Geologist field validation |

API documentation:

```text
http://127.0.0.1:8000/docs
```

# Streamlit Dashboard

**File:** `dashboard.py`

The dashboard provides:

### Top-level KPIs

- Potential ore
- Potential Mn
- Next-day production forecast
- Production target
- Expected shortfall
- Shortfall risk

### Geospatial view

- Prospectivity heatmap
- Study-area boundary
- Drill-hole locations
- Drill-hole Mn status, grade and depth
- Geologist field overrides
- Selectable base-map layers

### Decision support

- Shortfall risk
- Recommended actions
- Risk-driver breakdown

### Production analytics

- Historical production
- Production target
- Next-day AI prediction

### Operational audit

- Environmental conditions
- Equipment status
- Blasting status
- Model and pipeline audit

### Human-in-the-loop

Geologists can submit field observations directly from the dashboard. These observations are sent to FastAPI and stored as manual overrides.

# Project Structure

```text
sih26009_final_project/
│
├── code_SIH26009_final.ipynb
├── 02_finalize_prospectivity_model.py
├── app.py
├── dashboard.py
├── requirements.txt
├── README.md
│
├── synthetic_environmental.csv
├── synthetic_equipment.csv
├── synthetic_blasting.csv
├── synthetic_operational_features.csv
├── synthetic_production_targets.csv
├── synthetic_reserve_grid_enhanced.csv
│
└── outputs/
    ├── shortfall_predictions.csv
    ├── preventive_recommendations.csv
    ├── environment_zone_summary.csv
    ├── equipment_zone_summary.csv
    ├── blasting_zone_summary.csv
    ├── reserve_estimates_by_zone.csv
    ├── integrated_reserve_grid_predictions.csv
    ├── production_history_dashboard.csv
    ├── production_model.json
    ├── production_model_metrics.json
    ├── production_model_feature_importance.csv
    ├── study_area_bounds.json
    └── prospectivity model artifacts
```

# Installation

## Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd sih26009_final_project
```

## Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

# Run the Project

## Step 1 — Generate model artifacts

Open:

```text
code_SIH26009_final.ipynb
```

Run all cells.

To regenerate the prospectivity model and grid outputs:

```bash
python 02_finalize_prospectivity_model.py
```

## Step 2 — Start FastAPI

```bash
uvicorn app:app --reload --port 8000
```

Check:

```text
http://127.0.0.1:8000/health
```

or:

```text
http://127.0.0.1:8000/docs
```

## Step 3 — Start Streamlit

Open a second terminal in the same project folder:

```bash
streamlit run dashboard.py
```

Then open:

```text
http://localhost:8501
```

# Runtime Architecture

```text
                 CSV / Model Artifacts
                         │
                         ▼
                code_SIH26009_final.ipynb
                         │
                         ▼
                       outputs/
                         │
                         ▼
                    ┌───────────┐
                    │  FastAPI  │
                    │  app.py   │
                    └─────┬─────┘
                          │ REST API
                          ▼
                   ┌──────────────┐
                   │   Streamlit  │
                   │ dashboard.py │
                   └──────────────┘
                          │
                          ▼
                  Manager Decision View
```

# Prototype Data

The supplied demonstration datasets are synthetic and are intended to demonstrate the SIH26009 workflow and architecture.

For deployment, the synthetic datasets should be replaced with validated geological, production, equipment, blasting, weather and remote-sensing data.

# Future Improvements

1. Integrate real MOIL geological and operational data.
2. Add live satellite and weather data ingestion.
3. Improve spatial and temporal data synchronization.
4. Calibrate prospectivity scores and connect them with formal geological resource workflows.
5. Implement genuine multi-step production forecasting.
6. Add SHAP or similar model explainability.
7. Retrain using validated geologist overrides.
8. Add mine-schedule optimization and equipment allocation optimization.
9. Replace CSV persistence with a database.
10. Add authentication, logging, monitoring and production deployment.

