# ETL Pipeline — Data Science Salaries

A Python ETL pipeline that extracts, transforms, and loads a real-world 
data science salaries dataset, applying data cleaning, normalization, 
and feature engineering techniques.

## Dataset

Source: `ds_salaries.csv` — 607 records of data science professionals 
worldwide, including job title, experience level, salary in USD, 
remote ratio, and company size.

## Pipeline Overview

| Step | Operation | Technique |
|------|-----------|-----------|
| T1 | Remove redundant index column | `df.drop()` |
| T2 | Remove duplicate records | `drop_duplicates()` |
| T3 | Handle missing values | Median / Mode imputation |
| T4 | Standardize text fields | `str.lower().str.strip()` |
| T5 | Decode categorical variables | `map()` with lookup dict |
| T6 | Handle outliers | IQR clipping |
| T7 | Min-Max normalization | `MinMaxScaler` |
| T8 | Z-score standardization | `StandardScaler` |
| T9 | Feature engineering | `apply()` + `lambda` |
| T10 | Aggregation by experience | `groupby().agg()` |

## Output

- `Output/ds_salaries_refined.csv` — cleaned and enriched dataset 
  (565 records, 15 columns)
- `Output/resumen_por_experiencia.csv` — salary statistics grouped 
  by experience level

## Requirements
```bash
pip install pandas scikit-learn
```

## Usage

Place `ds_salaries.csv` inside the `Input/` folder, then run:
```bash
python Main.py
```

## Tech Stack

- Python 3.13
- Pandas
- Scikit-Learn