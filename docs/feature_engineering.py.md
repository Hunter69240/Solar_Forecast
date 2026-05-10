## feature_engineering.py

### Purpose
Constructs time-based and lag features from cleaned data for model training.

### What it does
1. Reads `data/cleaned.csv` with datetime as index
2. Extracts time-based features from the datetime index:
   - `hour` — hour of day (0–23)
   - `day_of_week` — day of week (0=Monday, 6=Sunday)
   - `month` — month of year (1–12)
3. Creates lag features from AC_POWER:
   - `lag_1` — AC_POWER value from 1 hour ago
   - `rolling_mean_3` — average AC_POWER over the past 3 hours
4. Drops rows with null values (first few rows have no lag history)
5. Saves result to `data/features.csv`

### Inputs
| File | Location |
|------|----------|
| cleaned.csv | data/ |

### Outputs
| File | Location | Description |
|------|----------|-------------|
| features.csv | data/ | Cleaned data with all features added |

### Key note
Lag features tell the model what happened recently — critical for LSTM 
which learns from sequences. Without these, the model only sees 
static time values and cannot learn temporal patterns.

### Dependencies
- `pandas`