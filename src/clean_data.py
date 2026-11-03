"""Step 1 - Data preprocessing: missing values, duplicates, outliers, feature + target creation."""
import pandas as pd
from common import *

NUMERIC = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No", "Total_Floors",
           "Nearby_Schools", "Nearby_Hospitals", "Parking_Space"]
log = []

df = pd.read_csv(RAW)
df.columns = df.columns.str.strip()
log.append(f"Raw rows: {len(df):,}  columns: {df.shape[1]}")
for c in df.select_dtypes(exclude="number").columns:
    df[c] = df[c].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

n = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
log.append(f"Duplicates removed: {n - len(df):,}")

# Parking_Space is Yes/No in this dataset -> 1/0 (numeric counts are kept as they are)
if df["Parking_Space"].astype(str).str.lower().isin(["yes", "no"]).any():
    df["Parking_Space"] = df["Parking_Space"].astype(str).str.lower().map({"yes": 1, "no": 0})
for c in NUMERIC:
    df[c] = pd.to_numeric(df[c], errors="coerce")
_m = df.isna().sum()[lambda s: s > 0]; log.append("Missing values before: " + ("none" if _m.empty else "\n" + _m.to_string()))
for c in NUMERIC:
    df[c] = df[c].fillna(df[c].median())
df["Amenities"] = df["Amenities"].fillna("")
for c in df.select_dtypes(exclude="number").columns:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

n = len(df)
df = df[(df["Size_in_SqFt"] > 0) & (df["Price_in_Lakhs"] > 0)]
log.append(f"Rows dropped for non-positive size/price: {n - len(df):,}")

def iqr_flag(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return (s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)
df["Outlier_Price"] = iqr_flag(df["Price_in_Lakhs"])
df["Outlier_Size"] = iqr_flag(df["Size_in_SqFt"])
log.append(f"IQR outliers flagged - price: {df.Outlier_Price.sum():,}, size: {df.Outlier_Size.sum():,}")
for c in ["Price_in_Lakhs", "Size_in_SqFt"]:          # winsorise at 1st / 99th percentile
    lo, hi = df[c].quantile([.01, .99]); df[c] = df[c].clip(lo, hi)

df = add_features(df)
df["Transport_Score"] = df["Transport_Score"].fillna(df["Transport_Score"].median())
df["Infra_Score"] = df["Infra_Score"].fillna(df["Infra_Score"].median())
df = add_targets(df)
df["Locality_Key"] = df["City"] + " - " + df["Locality"].astype(str)

df.to_csv(CLEAN, index=False)
log.append(f"Clean rows: {len(df):,}\nGood_Investment rate: {df.Good_Investment.mean():.1%}")
(OUT / "cleaning_report.txt").write_text("\n\n".join(log))
print("\n\n".join(log), f"\nSaved -> {CLEAN}")
