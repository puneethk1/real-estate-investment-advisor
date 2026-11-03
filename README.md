# Real Estate Investment Advisor
Predicts whether a property is a **Good Investment** (classification) and its **price after 5 years** (regression),
with a Streamlit app, EDA covering all 20 project questions, and MLflow experiment tracking.
Full methodology, findings and limitations: `docs/Project_Documentation.docx`.

## Run it locally
```bash
pip install -r requirements-train.txt     # use requirements.txt if you only want the app
# 1. Put india_housing_prices.csv in data/raw/   (not included in the zip)
python src/clean_data.py     # -> data/processed/cleaned.csv + outputs/cleaning_report.txt
python src/eda.py            # -> outputs/eda/*.png + outputs/eda_findings.md  (answers Q1-Q20)
python src/train.py          # -> models/*.joblib, outputs/model/*, MLflow runs in mlflow.db
streamlit run app.py         # the app
mlflow ui --backend-store-uri sqlite:///mlflow.db    # compare experiments
```
The zip already contains the results of these steps, so `streamlit run app.py` works straight away.
`python src/train.py --max-rows 250000` trains on every row (default is a 50,000-row sample for speed).

## Publish to GitHub
1. Create a free account at github.com and a **new empty repository** (e.g. `real-estate-investment-advisor`). Public is simplest for Streamlit Cloud.
2. In the unzipped project folder run (replace the URL with yours):
```bash
git init
git add .
git commit -m "Real Estate Investment Advisor"
git branch -M main
git remote add origin https://github.com/<your-username>/real-estate-investment-advisor.git
git push -u origin main
```
3. **File size:** `data/processed/cleaned.csv` is about 72 MB. GitHub blocks files over 100 MB and only warns above 50 MB, so a normal push works.
   If it is ever rejected or the app runs out of memory, shrink it first and push again:
   `gzip data/processed/cleaned.csv` (about 20 MB). The code reads `cleaned.csv.gz` automatically when `cleaned.csv` is missing.
4. Do **not** commit the raw dataset (it is already in `.gitignore`).

## Deploy on Streamlit Community Cloud
1. Go to share.streamlit.io and sign in with GitHub, then choose **Create app** -> *Deploy a public app from GitHub*.
2. Pick your repository, branch `main`, and main file path `app.py`.
3. Open **Advanced settings** and choose Python **3.12**, then click **Deploy**. The first build takes a few minutes.
4. The repo must contain `models/classifier.joblib`, `models/regressor.joblib`, `outputs/` and `data/processed/cleaned.csv` (or `.csv.gz`).
   `requirements.txt` pins the library versions the models were trained with; keep them in sync if you retrain on a different version.
5. Copy the app URL (`https://<name>.streamlit.app`) into your documentation and submission.

If the app shows "Run the pipeline first", a file listed in step 4 was not committed. If it crashes on load, use the `.csv.gz` option above.

## Layout
| Path | Purpose |
|---|---|
| `src/common.py` | paths, feature lists, feature engineering, target creation |
| `src/clean_data.py` | missing values, duplicates, outliers, features, targets |
| `src/eda.py` | the 20 EDA questions -> charts + written findings |
| `src/train.py` | 3 classifiers + 3 regressors, metrics, MLflow logging, saves best models |
| `app.py` | Streamlit: Predict / Explore listings / Market insights / Model performance |
| `docs/Project_Documentation.docx` | methodology, findings, use cases, limitations |
| `tests/make_sample_data.py` | SYNTHETIC data to smoke-test the pipeline only - never report its results |
