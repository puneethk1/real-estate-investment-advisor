"""Shared paths, feature lists and feature-engineering used by every script and the app."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "india_housing_prices.csv"
CLEAN = ROOT / "data" / "processed" / "cleaned.csv"
if not CLEAN.exists() and CLEAN.with_suffix(".csv.gz").exists():   # optional smaller copy: gzip data/processed/cleaned.csv
    CLEAN = CLEAN.with_suffix(".csv.gz")
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
for _p in (RAW.parent, CLEAN.parent, MODELS, OUT / "eda", OUT / "model"):
    _p.mkdir(parents=True, exist_ok=True)

HORIZON = 5  # years
NUM_FEATURES = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
                "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Transport_Score",
                "Parking_Space", "Amenity_Count", "Floor_Ratio", "School_Density_Score", "Infra_Score"]
CAT_FEATURES = ["State", "City", "Property_Type", "Furnished_Status", "Facing", "Owner_Type",
                "Availability_Status", "Security"]


def to_transport(s: pd.Series) -> pd.Series:
    """Public transport accessibility may be numeric (1-10) or Low/Medium/High text."""
    numeric = pd.to_numeric(s, errors="coerce")
    mapped = s.astype(str).str.lower().str.strip().map({"low": 3, "medium": 6, "high": 9})
    return numeric.fillna(mapped)


def amenity_count(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).apply(
        lambda x: len([a for a in x.replace(";", ",").split(",") if a.strip()]))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features. Used in cleaning AND at prediction time in the app."""
    df = df.copy()
    df["Age_of_Property"] = pd.Timestamp.today().year - pd.to_numeric(df["Year_Built"], errors="coerce")
    df["Transport_Score"] = to_transport(df["Public_Transport_Accessibility"])
    df["Amenity_Count"] = amenity_count(df["Amenities"])
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]          # Rs per sq ft
    df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"].replace(0, np.nan)).fillna(0)
    df["School_Density_Score"] = df["Nearby_Schools"] / df["Size_in_SqFt"] * 1000
    df["Infra_Score"] = (df["Nearby_Schools"] + df["Nearby_Hospitals"] + df["Transport_Score"]) / 3
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create the two target variables (rule-based, see DOCUMENTATION.md)."""
    df = df.copy()
    med = lambda c: df[c].median()
    city_pps = df.groupby("City")["Price_per_SqFt"].transform("median")
    ready = df["Availability_Status"].astype(str).str.lower().str.contains("ready|available")
    checks = [df["BHK"] >= 3, df["Transport_Score"] >= med("Transport_Score"),
              df["Nearby_Schools"] >= med("Nearby_Schools"), df["Nearby_Hospitals"] >= med("Nearby_Hospitals"),
              df["Age_of_Property"] <= med("Age_of_Property"), ready]
    df["Investment_Score"] = sum(c.astype(int) for c in checks)                    # 0-6
    df["Good_Investment"] = ((df["Price_per_SqFt"] <= city_pps) & (df["Investment_Score"] >= 3)).astype(int)
    # Growth rate: 6-10% depending on how expensive/hot the city is, +0.5% each for good transport / infra
    city_rank = city_pps.rank(pct=True)
    df["Growth_Rate"] = 0.06 + 0.04 * city_rank + 0.005 * (df["Transport_Score"] >= med("Transport_Score")) \
        + 0.005 * (df["Infra_Score"] >= med("Infra_Score"))
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** HORIZON
    return df
