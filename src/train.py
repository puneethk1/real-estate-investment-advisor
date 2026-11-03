"""Step 3 - Model training + evaluation (classification & regression) with MLflow tracking.
Usage: python src/train.py [--max-rows 80000]"""
import argparse, contextlib, json, warnings
import joblib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import *

warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None
try:
    import mlflow, mlflow.sklearn
except ImportError:
    mlflow = None

ap = argparse.ArgumentParser(); ap.add_argument("--max-rows", type=int, default=50000)
args = ap.parse_args()
df = pd.read_csv(CLEAN)
if len(df) > args.max_rows:
    df = df.sample(args.max_rows, random_state=42)
X = df[NUM_FEATURES + CAT_FEATURES]
if mlflow:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    EXP = "real-estate-investment-advisor"
    if mlflow.get_experiment_by_name(EXP) is None:
        mlflow.create_experiment(EXP, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXP)

def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM_FEATURES),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)])

@contextlib.contextmanager
def track(name):
    if mlflow:
        with mlflow.start_run(run_name=name) as r:
            yield r
    else:
        yield None

def importances(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out(); m = pipe.named_steps["model"]
    vals = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(np.ravel(m.coef_))
    agg = {}
    for n, v in zip(names, vals):
        base = n.split("__", 1)[1]
        for c in CAT_FEATURES:
            if base.startswith(c + "_"):
                base = c; break
        agg[base] = agg.get(base, 0) + float(v)
    s = pd.Series(agg).sort_values(ascending=False)
    return s / s.sum()

def fit_all(task, models, Xtr, Xte, ytr, yte, metric_fn, key, higher_better):
    res, fitted = {}, {}
    for name, est in models.items():
        pipe = Pipeline([("prep", prep()), ("model", est)]).fit(Xtr, ytr)
        m = metric_fn(pipe, Xte, yte)
        with track(f"{task}-{name}") as run:
            if run:
                mlflow.log_params({"task": task, "model": name, "n_train": len(Xtr), "n_features": X.shape[1]})
                mlflow.log_metrics(m)
                try:
                    mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
                except TypeError:
                    mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="cloudpickle")
        res[name], fitted[name] = m, pipe
        print(f"[{task}] {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    best = (max if higher_better else min)(res, key=lambda k: res[k][key])
    return res, fitted, best

# ---------------- classification: Good_Investment ----------------
y = df["Good_Investment"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
def clf_metrics(p, Xt, yt):
    pr, pp = p.predict(Xt), p.predict_proba(Xt)[:, 1]
    return {"accuracy": accuracy_score(yt, pr), "precision": precision_score(yt, pr), "recall": recall_score(yt, pr),
            "f1": f1_score(yt, pr), "roc_auc": roc_auc_score(yt, pp)}
clf_models = {"logistic_regression": LogisticRegression(max_iter=1000),
              "random_forest": RandomForestClassifier(n_estimators=40, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=42)}
if XGBClassifier:
    clf_models["xgboost"] = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=.1, n_jobs=-1, eval_metric="logloss")
c_res, c_fit, c_best = fit_all("classification", clf_models, Xtr, Xte, ytr, yte, clf_metrics, "f1", True)
ConfusionMatrixDisplay(confusion_matrix(yte, c_fit[c_best].predict(Xte)), display_labels=["Not good", "Good"]).plot(cmap="Blues")
plt.title(f"Confusion matrix - {c_best}"); plt.savefig(OUT / "model" / "confusion_matrix.png", dpi=110); plt.close()
importances(c_fit[c_best]).to_csv(OUT / "model" / "clf_importance.csv", header=["importance"])
joblib.dump(c_fit[c_best], MODELS / "classifier.joblib")

# ---------------- regression: price after 5 years ----------------
y = df["Future_Price_5Y"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
def reg_metrics(p, Xt, yt):
    pr = p.predict(Xt)
    return {"rmse": float(np.sqrt(mean_squared_error(yt, pr))), "mae": mean_absolute_error(yt, pr), "r2": r2_score(yt, pr)}
reg_models = {"ridge": Ridge(alpha=1.0),
              "random_forest": RandomForestRegressor(n_estimators=40, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=42)}
if XGBRegressor:
    reg_models["xgboost"] = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=.1, n_jobs=-1)
r_res, r_fit, r_best = fit_all("regression", reg_models, Xtr, Xte, ytr, yte, reg_metrics, "rmse", False)
pred = r_fit[r_best].predict(Xte); k = min(2000, len(yte))
plt.scatter(yte.values[:k], pred[:k], s=6, alpha=.4); lim = [0, float(yte.max())]; plt.plot(lim, lim, "r--")
plt.xlabel("Actual 5-yr price (L)"); plt.ylabel("Predicted"); plt.title(f"Regression - {r_best}")
plt.savefig(OUT / "model" / "regression_actual_vs_pred.png", dpi=110); plt.close()
importances(r_fit[r_best]).to_csv(OUT / "model" / "reg_importance.csv", header=["importance"])
joblib.dump(r_fit[r_best], MODELS / "regressor.joblib")

(OUT / "model" / "metrics.json").write_text(json.dumps(
    {"classification": {"best": c_best, "results": c_res}, "regression": {"best": r_best, "results": r_res},
     "train_rows": len(df)}, indent=2))
print(f"\nBest classifier: {c_best} | best regressor: {r_best}. Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to compare runs.")
