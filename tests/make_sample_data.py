"""Generates a SYNTHETIC file with the same schema, ONLY to smoke-test the pipeline. Never use for results."""
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(0); n = 6000
cities = {"Karnataka": ["Bangalore", "Mysore"], "Maharashtra": ["Mumbai", "Pune"], "Tamil Nadu": ["Chennai", "Coimbatore"]}
st = rng.choice(list(cities), n); city = [rng.choice(cities[s]) for s in st]
size = rng.integers(500, 4500, n); base = pd.Series(city).map({"Mumbai": 24, "Bangalore": 14, "Pune": 11, "Chennai": 10, "Mysore": 6, "Coimbatore": 5}).values
price = size * base / 1e3 * rng.uniform(.7, 1.4, n) * 10 / 10
am = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]
df = pd.DataFrame({"ID": range(n), "State": st, "City": city, "Locality": [f"Locality_{i}" for i in rng.integers(1, 40, n)],
    "Property_Type": rng.choice(["Apartment", "Villa", "Independent House"], n), "BHK": rng.integers(1, 6, n),
    "Size_in_SqFt": size, "Price_in_Lakhs": price.round(2), "Price_per_SqFt": 0.1, "Year_Built": rng.integers(1990, 2024, n),
    "Furnished_Status": rng.choice(["Unfurnished", "Semi-furnished", "Furnished"], n), "Floor_No": rng.integers(0, 20, n),
    "Total_Floors": rng.integers(1, 30, n), "Age_of_Property": 0, "Nearby_Schools": rng.integers(1, 11, n),
    "Nearby_Hospitals": rng.integers(1, 11, n), "Public_Transport_Accessibility": rng.choice(["Low", "Medium", "High"], n),
    "Parking_Space": rng.integers(0, 4, n), "Security": rng.choice(["Yes", "No"], n),
    "Amenities": [", ".join(rng.choice(am, rng.integers(0, 5), replace=False)) for _ in range(n)],
    "Facing": rng.choice(["North", "South", "East", "West"], n), "Owner_Type": rng.choice(["Owner", "Builder", "Broker"], n),
    "Availability_Status": rng.choice(["Ready_to_Move", "Under_Construction"], n)})
df.loc[rng.choice(n, 150, replace=False), "Nearby_Schools"] = np.nan          # missing values
df = pd.concat([df, df.sample(40, random_state=1)])                              # duplicates
Path(__file__).resolve().parents[1].joinpath("data/raw/india_housing_prices.csv").write_text(df.to_csv(index=False))
