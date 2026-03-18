"""
features.py

Purpose:
Creates new features (variables) that help explain cancellations.

What this file does:
- Creates total_nights (weekend + weekday stays)
- Creates total_guests (adults + children + babies)

Why it exists:
Feature engineering improves model performance.
Keeps transformation logic separate from cleaning and modeling.
"""

import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates extra variables that may help explain cancellations.

    Adds:
    - total_nights = staysinweeknights + staysinweekendnights
    - total_guests = adults + children + babies
    """
    df = df.copy()

    # total nights
    if "staysinweeknights" in df.columns and "staysinweekendnights" in df.columns:
        df["total_nights"] = df["staysinweeknights"] + df["staysinweekendnights"]
    else:
        df["total_nights"] = 0
    
    #checks if there are children and babies and returns 1 if a family, 0 if not
    if "children" in df.columns and "babies" in df.columns:
        df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
 
    #checks if existence of special requests as a binary 1 if it has more than 0, 0 if not
    if "totalofspecialrequests" in df.columns:
        df["has_special_requests"] = (df["totalofspecialrequests"] > 0).astype(int)

    # total guests
    needed = {"adults", "children", "babies"}
    if needed.issubset(df.columns):
        df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    else:
        df["total_guests"] = pd.NA

    return df