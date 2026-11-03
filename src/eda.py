"""Step 2 - EDA: answers the 20 project questions with a chart + a written finding each."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import *

df = pd.read_csv(CLEAN)
sns.set_theme(style="whitegrid")
F = []

def save(name):
    plt.tight_layout(); plt.savefig(OUT / "eda" / f"{name}.png", dpi=110); plt.close()

def note(q, a):
    F.append(f"### {q}\n\n{a}\n")

def binned(col, target="Price_per_SqFt"):
    x = df[col] if df[col].nunique() <= 12 else pd.qcut(df[col], 8, duplicates="drop")
    return df.groupby(x, observed=True)[target].mean()

def iqr_outliers(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())

# ---- 1-5 Price & size
sns.histplot(df.Price_in_Lakhs, bins=50, kde=True); plt.title("Price (lakhs)"); save("q01_price_dist")
note("1. Distribution of property prices", f"Mean {df.Price_in_Lakhs.mean():.1f}L, median {df.Price_in_Lakhs.median():.1f}L, "
     f"skew {df.Price_in_Lakhs.skew():.2f} (positive = a tail of expensive properties).")
sns.histplot(df.Size_in_SqFt, bins=50, kde=True); plt.title("Size (sq ft)"); save("q02_size_dist")
note("2. Distribution of property sizes", f"Mean {df.Size_in_SqFt.mean():.0f} sq ft, median {df.Size_in_SqFt.median():.0f}, "
     f"range {df.Size_in_SqFt.min():.0f}-{df.Size_in_SqFt.max():.0f}.")
sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt"); plt.title("Price/sqft by property type"); save("q03_pps_by_type")
m = df.groupby("Property_Type").Price_per_SqFt.median().sort_values()
note("3. Price per sq ft by property type", f"Highest median: {m.index[-1]} (Rs {m.iloc[-1]:,.0f}); lowest: {m.index[0]} (Rs {m.iloc[0]:,.0f}).")
s = df.sample(min(len(df), 5000), random_state=1)
sns.scatterplot(data=s, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=.3); plt.title("Size vs price"); save("q04_size_vs_price")
note("4. Size vs price", f"Correlation = {df.Size_in_SqFt.corr(df.Price_in_Lakhs):.2f}.")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df.Price_per_SqFt, ax=ax[0]); sns.boxplot(y=df.Size_in_SqFt, ax=ax[1]); save("q05_outliers")
note("5. Outliers", f"IQR outliers - price/sqft: {iqr_outliers(df.Price_per_SqFt):,}; size: {iqr_outliers(df.Size_in_SqFt):,}. "
     "Price and size were winsorised at 1%/99% during cleaning (raw flags kept in Outlier_* columns).")

# ---- 6-10 Location
g = df.groupby("State").Price_per_SqFt.mean().sort_values(ascending=False)
g.plot(kind="bar", figsize=(10, 4), title="Avg price/sqft by state"); save("q06_pps_by_state")
note("6. Average price/sqft by state", f"Highest: {g.index[0]} (Rs {g.iloc[0]:,.0f}); lowest: {g.index[-1]} (Rs {g.iloc[-1]:,.0f}).")
g = df.groupby("City").Price_in_Lakhs.mean().sort_values(ascending=False)
g.head(25).plot(kind="bar", figsize=(11, 4), title="Avg price (lakhs) by city (top 25)"); save("q07_price_by_city")
note("7. Average price by city", f"Most expensive city: {g.index[0]} ({g.iloc[0]:.1f}L); cheapest: {g.index[-1]} ({g.iloc[-1]:.1f}L).")
g = df.groupby("Locality_Key").Age_of_Property.median().sort_values()
g.to_csv(OUT / "eda" / "q08_median_age_by_locality.csv")
g.tail(15).plot(kind="barh", figsize=(8, 5), title="Oldest localities (median age, yrs)"); save("q08_age_by_locality")
note("8. Median age by locality", f"Overall median {df.Age_of_Property.median():.0f} yrs; oldest locality {g.index[-1]} ({g.iloc[-1]:.0f} yrs). Full table in q08 CSV.")
ct = pd.crosstab(df.City, df.BHK, normalize="index")
plt.figure(figsize=(8, max(4, len(ct) * .25))); sns.heatmap(ct, cmap="YlGnBu"); plt.title("BHK mix by city"); save("q09_bhk_by_city")
note("9. BHK distribution across cities", f"Most common BHK overall: {int(df.BHK.mode()[0])}. See heatmap for city-level mix.")
top5 = df.groupby("Locality_Key").Price_in_Lakhs.mean().nlargest(5).index
sub = df[df.Locality_Key.isin(top5)].assign(Built=lambda d: d.Year_Built // 5 * 5)
sub.groupby(["Built", "Locality_Key"]).Price_per_SqFt.mean().unstack().plot(marker="o", figsize=(9, 4), title="Price/sqft trend, 5 priciest localities (by year built)")
save("q10_top5_locality_trend")
note("10. Price trend, top-5 priciest localities", "Priciest: " + ", ".join(top5) + ". The dataset has no sale dates, so the trend is price/sqft by construction year.")

# ---- 11-15 Relationships
num = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Nearby_Schools",
       "Nearby_Hospitals", "Transport_Score", "Parking_Space", "Amenity_Count"]
c = df[num].corr()
plt.figure(figsize=(9, 7)); sns.heatmap(c, annot=True, fmt=".2f", cmap="coolwarm", center=0); save("q11_correlation")
p = c.where(~np.eye(len(c), dtype=bool)).abs().stack().idxmax()
note("11. Numeric correlations", f"Strongest pair: {p[0]} & {p[1]} (r = {c.loc[p]:.2f}).")
for q, col, nm in [(12, "Nearby_Schools", "schools"), (13, "Nearby_Hospitals", "hospitals")]:
    b = binned(col); b.plot(marker="o", figsize=(7, 4), title=f"Avg price/sqft vs {col}"); save(f"q{q}_{nm}_vs_pps")
    note(f"{q}. {col} vs price/sqft", f"Correlation = {df[col].corr(df.Price_per_SqFt):.2f}.")
sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs"); save("q14_price_by_furnishing")
m = df.groupby("Furnished_Status").Price_in_Lakhs.median()
note("14. Price by furnished status", "Median price (lakhs): " + ", ".join(f"{k} {v:.1f}" for k, v in m.items()) + ".")
g = df.groupby("Facing").Price_per_SqFt.mean().sort_values()
g.plot(kind="bar", title="Avg price/sqft by facing"); save("q15_pps_by_facing")
note("15. Price/sqft by facing", f"Highest: {g.index[-1]} (Rs {g.iloc[-1]:,.0f}); lowest: {g.index[0]} (Rs {g.iloc[0]:,.0f}); "
     f"spread {(g.iloc[-1] / g.iloc[0] - 1):.1%}.")

# ---- 16-20 Ownership / amenities / investment
for q, col, nm in [(16, "Owner_Type", "owners"), (17, "Availability_Status", "availability")]:
    v = df[col].value_counts(); v.plot(kind="bar", title=f"Properties by {col}"); save(f"q{q}_{nm}")
    note(f"{q}. Count by {col}", ", ".join(f"{k}: {n:,}" for k, n in v.items()) + ".")
b = df.groupby("Parking_Space").Price_in_Lakhs.mean(); b.plot(kind="bar", title="Avg price by parking (0 = No, 1 = Yes)"); save("q18_parking_vs_price")
note("18. Parking and price", f"Correlation = {df.Parking_Space.corr(df.Price_in_Lakhs):.2f}. Avg price by parking, 0=No/1=Yes (lakhs): " + ", ".join(f"{k}: {v:.1f}" for k, v in b.items()) + ".")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df, x="Amenity_Count", y="Price_per_SqFt", ax=ax[0])
ex = df.assign(A=df.Amenities.str.split(",")).explode("A"); ex["A"] = ex.A.str.strip()
ex = ex[ex.A != ""]; top = ex.groupby("A").Price_per_SqFt.mean().sort_values(); top.plot(kind="barh", ax=ax[1], title="Avg price/sqft by amenity"); save("q19_amenities")
note("19. Amenities and price/sqft", f"Correlation of amenity count with price/sqft = {df.Amenity_Count.corr(df.Price_per_SqFt):.2f}. "
     + (f"Amenity with highest avg price/sqft: {top.index[-1]}." if len(top) else ""))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
binned("Transport_Score").plot(marker="o", ax=ax[0], title="Price/sqft vs transport score")
binned("Transport_Score", "Good_Investment").plot(marker="o", color="green", ax=ax[1], title="Good-investment rate vs transport score"); save("q20_transport")
note("20. Public transport vs price/sqft and investment potential", f"Correlation with price/sqft = {df.Transport_Score.corr(df.Price_per_SqFt):.2f}; "
     f"with Good_Investment = {df.Transport_Score.corr(df.Good_Investment):.2f}.")

# ---- extra: investment-focused views from the Data Analyst brief
g = df.groupby("City").Good_Investment.mean().sort_values(ascending=False)
g.head(25).plot(kind="bar", figsize=(11, 4), title="Share of good investments by city"); save("x1_good_by_city")
note("Extra: Good-investment share by city", f"Highest: {g.index[0]} ({g.iloc[0]:.0%}); lowest: {g.index[-1]} ({g.iloc[-1]:.0%}).")
note("Extra: Area vs return", f"Correlation of size with implied 5-yr growth rate = {df.Size_in_SqFt.corr(df.Growth_Rate):.2f}.")

(OUT / "eda_findings.md").write_text("# EDA findings (auto-generated)\n\n" + "\n".join(F))
print(f"{len(F)} findings written to outputs/eda_findings.md; charts in outputs/eda/")
