import pandas as pd
import json
import os

df = pd.read_csv('data/raw/cbvd-5/CBVD-5.csv')
num_rows = len(df)
columns = list(df.columns)
unique_behavior_ids = df['class'].unique().tolist() if 'class' in df.columns else df.iloc[:, -1].unique().tolist() # guess

report = f"""# CBVD-5 Actual Annotation Verification

## File Details
- **File:** `CBVD-5.csv`
- **Total Rows:** {num_rows}
- **Columns:** {columns}

## Class Information
- **Unique Behaviour IDs:** {unique_behavior_ids}
"""

print(report)

print("First few rows:")
print(df.head())
