"""
cleaning.py

Purpose:
Cleans the raw hotel booking data before modeling.

What this file does:
- Handles missing values
- Fixes incorrect data types
- Removes invalid rows if necessary

Why it exists:
Keeps preprocessing separate from modeling.
Makes the project modular and clean.
"""

import pandas as pd

def clean_hotel_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Make column names consistent
    df.columns = df.columns.str.lower()

    # Fix missing values
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)
    
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unnamed")

    # Ensure target exists
    if "iscanceled" not in df.columns:
        raise KeyError("Expected column 'IsCanceled' (which becomes 'iscanceled' after lowercasing).")

    # Convert target to int
    df["iscanceled"] = df["iscanceled"].astype(int)

    # Remove rows missing adr and remove negative adr values
    if "adr" in df.columns:
        df = df[df["adr"].notna() & (df["adr"] >= 0)]

    return df