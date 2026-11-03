"""Step 4 - Streamlit app: prediction + market insights.  Run: streamlit run app.py"""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import *

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(CLEAN)

@st.cache_resource
def load_models():
    return joblib.load(MODELS / "classifier.joblib"), joblib.load(MODELS / "regressor.joblib")

if not CLEAN.exists() or not (MODELS / "classifier.joblib").exists():
    st.error("Run the pipeline first: python src/clean_data.py && python src/eda.py && python src/train.py")
    st.stop()
df, (clf, reg) = load_data(), load_models()
metrics = json.loads((OUT / "model" / "metrics.json").read_text())

st.title("🏠 Real Estate Investment Advisor")
st.caption("Is this property a good investment, and what could it be worth in 5 years?")
t1, t2, t3, t4 = st.tabs(["🔮 Predict", "🔎 Explore listings", "📊 Market insights", "🧪 Model performance"])

# ------------------------------------------------------------------ Predict
with t1:
    d = df
    a, b, c = st.columns(3)
    with a:
        state = st.selectbox("State", sorted(d.State.unique()))
        city = st.selectbox("City", sorted(d[d.State == state].City.unique()))
        ptype = st.selectbox("Property type", sorted(d.Property_Type.unique()))
        furn = st.selectbox("Furnished status", sorted(d.Furnished_Status.unique()))
        facing = st.selectbox("Facing", sorted(d.Facing.unique()))
    with b:
        bhk = st.number_input("BHK", 1, int(d.BHK.max()), int(d.BHK.median()))
        size = st.number_input("Size (sq ft)", 100, 20000, int(d.Size_in_SqFt.median()), step=50)
        price = st.number_input("Current price (lakhs)", 1.0, float(d.Price_in_Lakhs.max()) * 2, float(round(d.Price_in_Lakhs.median(), 1)))
        year = st.number_input("Year built", int(d.Year_Built.min()), pd.Timestamp.today().year, int(d.Year_Built.median()))
        floor = st.number_input("Floor no.", 0, 100, int(d.Floor_No.median()))
        total = st.number_input("Total floors", 1, 100, max(int(d.Total_Floors.median()), 1))
    with c:
        sch = st.slider("Nearby schools", int(d.Nearby_Schools.min()), int(d.Nearby_Schools.max()), int(d.Nearby_Schools.median()))
        hos = st.slider("Nearby hospitals", int(d.Nearby_Hospitals.min()), int(d.Nearby_Hospitals.max()), int(d.Nearby_Hospitals.median()))
        trn = st.slider("Public transport accessibility", int(d.Transport_Score.min()), int(d.Transport_Score.max()), int(d.Transport_Score.median()))
        park = 1 if st.selectbox("Parking space", ["No", "Yes"], index=int(d.Parking_Space.median())) == "Yes" else 0
        sec = st.selectbox("Security", sorted(d.Security.unique()))
        owner = st.selectbox("Owner type", sorted(d.Owner_Type.unique()))
        avail = st.selectbox("Availability", sorted(d.Availability_Status.unique()))
    opts = sorted({x.strip() for v in d.Amenities.dropna() for x in str(v).split(",") if x.strip()})
    amen = st.multiselect("Amenities", opts)

    if st.button("Analyse property", type="primary"):
        row = add_features(pd.DataFrame([{
            "State": state, "City": city, "Property_Type": ptype, "Furnished_Status": furn, "Facing": facing, "BHK": bhk,
            "Size_in_SqFt": size, "Price_in_Lakhs": price, "Year_Built": year, "Floor_No": floor, "Total_Floors": total,
            "Nearby_Schools": sch, "Nearby_Hospitals": hos, "Public_Transport_Accessibility": trn, "Parking_Space": park,
            "Security": sec, "Owner_Type": owner, "Availability_Status": avail, "Amenities": ", ".join(amen)}]))
        X = row[NUM_FEATURES + CAT_FEATURES]
        p_good = float(clf.predict_proba(X)[0, 1]); future = float(reg.predict(X)[0])
        cagr = (future / price) ** (1 / HORIZON) - 1
        m1, m2, m3 = st.columns(3)
        m1.metric("Good investment?", "✅ Yes" if p_good >= .5 else "❌ No")
        m2.metric("Model confidence", f"{max(p_good, 1 - p_good):.0%}", help=f"P(good investment) = {p_good:.0%}")
        m3.metric(f"Estimated price in {HORIZON} yrs", f"₹{future:,.1f} L", f"{future - price:+,.1f} L ({cagr:.1%}/yr)")
        city_pps = d[d.City == city].Price_per_SqFt.median(); this_pps = float(row.Price_per_SqFt.iloc[0])
        st.info(f"This property is ₹{this_pps:,.0f}/sq ft vs a {city} median of ₹{city_pps:,.0f}/sq ft "
                f"({this_pps / city_pps - 1:+.0%}).")
        st.progress(p_good, text=f"P(good investment) = {p_good:.0%}")

    st.subheader("What drives the predictions")
    i1, i2 = st.columns(2)
    for col, f, ttl in [(i1, "clf_importance.csv", "Good-investment model"), (i2, "reg_importance.csv", "Price-forecast model")]:
        imp = pd.read_csv(OUT / "model" / f, index_col=0).head(10)["importance"].sort_values()
        col.plotly_chart(px.bar(imp, orientation="h", title=f"Feature importance - {ttl}").update_layout(showlegend=False, xaxis_title="", yaxis_title=""), width="stretch")

# ------------------------------------------------------------------ Explore
with t2:
    f1, f2, f3, f4 = st.columns(4)
    states = f1.multiselect("State", sorted(df.State.unique()))
    cities = f2.multiselect("City", sorted(df[df.State.isin(states)].City.unique()) if states else sorted(df.City.unique()))
    bhks = f3.multiselect("BHK", sorted(df.BHK.unique()))
    only_good = f4.checkbox("Only good investments")
    lo, hi = float(df.Price_in_Lakhs.min()), float(df.Price_in_Lakhs.max())
    pr = st.slider("Price (lakhs)", lo, hi, (lo, hi))
    sz = st.slider("Size (sq ft)", int(df.Size_in_SqFt.min()), int(df.Size_in_SqFt.max()), (int(df.Size_in_SqFt.min()), int(df.Size_in_SqFt.max())))
    v = df[df.Price_in_Lakhs.between(*pr) & df.Size_in_SqFt.between(*sz)]
    if states: v = v[v.State.isin(states)]
    if cities: v = v[v.City.isin(cities)]
    if bhks: v = v[v.BHK.isin(bhks)]
    if only_good: v = v[v.Good_Investment == 1]
    st.write(f"**{len(v):,}** properties match")
    cols = ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
            "Availability_Status", "Good_Investment", "Future_Price_5Y"]
    st.dataframe(v[[c for c in cols if c in v]].head(500), width="stretch")
    st.download_button("Download filtered data (CSV)", v.to_csv(index=False), "filtered_properties.csv")
    if len(v):
        st.plotly_chart(px.scatter(v.sample(min(len(v), 3000), random_state=1), x="Size_in_SqFt", y="Price_in_Lakhs",
                                   color="Good_Investment", hover_data=["City", "Locality"], opacity=.6), width="stretch")

# ------------------------------------------------------------------ Insights
with t3:
    h = df.pivot_table(index="City", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean")
    st.plotly_chart(px.imshow(h, aspect="auto", color_continuous_scale="YlOrRd", title="Avg price/sq ft: city × property type (heatmap)"), width="stretch")
    a, b = st.columns(2)
    g = df.groupby("City").Good_Investment.mean().sort_values(ascending=False).head(20)
    a.plotly_chart(px.bar(g, title="Share of good investments by city (top 20)").update_layout(showlegend=False), width="stretch")
    sel = b.selectbox("City for price trend", sorted(df.City.unique()))
    t = df[df.City == sel].assign(Built=lambda x: x.Year_Built // 5 * 5).groupby("Built")[["Price_per_SqFt", "Future_Price_5Y", "Price_in_Lakhs"]].mean().reset_index()
    b.plotly_chart(px.line(t, x="Built", y="Price_per_SqFt", markers=True, title=f"{sel}: avg price/sq ft by construction period"), width="stretch")
    num = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Transport_Score", "Parking_Space", "Amenity_Count"]
    st.plotly_chart(px.imshow(df[num].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation of numeric features"), width="stretch")
    with st.expander("All 20 EDA charts + written findings"):
        for img in sorted((OUT / "eda").glob("*.png")):
            st.image(str(img), caption=img.stem)
        if (OUT / "eda_findings.md").exists():
            st.markdown((OUT / "eda_findings.md").read_text())

# ------------------------------------------------------------------ Model performance
with t4:
    for task, cols in [("classification", ["accuracy", "precision", "recall", "f1", "roc_auc"]), ("regression", ["rmse", "mae", "r2"])]:
        st.subheader(f"{task.title()} — best: {metrics[task]['best']}")
        st.dataframe(pd.DataFrame(metrics[task]["results"]).T[cols].round(3), width="stretch")
    a, b = st.columns(2)
    a.image(str(OUT / "model" / "confusion_matrix.png")); b.image(str(OUT / "model" / "regression_actual_vs_pred.png"))
    st.caption("Experiments are logged in MLflow: run `mlflow ui --backend-store-uri sqlite:///mlflow.db`.")
