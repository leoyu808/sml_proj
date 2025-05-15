import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
import xgboost as xgb
from xgboost import XGBClassifier
import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import shap

TEAMS = [100, 200]

def load_data(split):
    pattern = f"/scratch/network/ly4431/wpe/wpe_{split}_part*.csv"
    files = glob.glob(pattern)

    def extract_index(fn: str) -> int:
        m = re.search(rf"wpe_{split}_part(\d+)\.csv$", fn)
        return int(m.group(1)) if m else -1

    files = sorted(files, key=extract_index)

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.drop(columns=['MATCH_ID'], errors='ignore')
    df = df.drop(columns=[f"{'BLUE' if team==100 else 'RED'}_{role}_{feat}"
    for team, (role, feat) in product(TEAMS, product(["TOP", "JG", "MID", "BOT", "SUP"], ["XP", "TOTAL_GOLD"]))], errors='ignore')
    return df

X = load_data("X")
y = load_data("Y")
y['WIN_TEAMID'] = y['WIN_TEAMID'].map({100: 0, 200: 1})
y = y['WIN_TEAMID']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def fit_logistic_regression(X_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver='saga',
            penalty='l1',
            max_iter=500,
            tol=1e-4,
            n_jobs=1,
            random_state=42,
            verbose=1
        )
    )
    model.fit(X_train, y_train)
    return model

def fit_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model

def fit_xgboost(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
    )

    model = XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        verbosity=1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=1
    )
    return model

model = fit_logistic_regression(X_train, y_train)

def evaluate_win_probabilities(y_true, y_pred_prob, threshold=0.5):
    auc    = roc_auc_score(y_true, y_pred_prob)
    ll     = log_loss(y_true, y_pred_prob)
    brier  = brier_score_loss(y_true, y_pred_prob)
    y_pred = (y_pred_prob >= threshold).astype(int)
    acc    = accuracy_score(y_true, y_pred)

    return {
        "roc_auc":    auc,
        "log_loss":   ll,
        "brier_score": brier,
        "accuracy":   acc
    }

def evaluate_by_interval(X_test, y_test, y_pred_prob, interval_minutes=5, threshold=0.5):
    df = pd.DataFrame({
        'timestamp':     X_test['TIMESTAMP'].values,
        'y_true':        y_test,
        'y_pred_prob':   y_pred_prob
    })
    interval_ms = interval_minutes * 60 * 1000
    df['interval_start'] = (df['timestamp'] // interval_ms) * interval_ms
    records = []
    for interval, group in df.groupby('interval_start', sort=True):
        metrics = evaluate_win_probabilities(
            group['y_true'], group['y_pred_prob'], threshold=threshold
        )
        metrics['interval_start'] = interval
        records.append(metrics)
    return pd.DataFrame.from_records(records).set_index('interval_start')

# y_pred = model.predict_proba(X_test[feature_cols])[:, 1]
# interval_metrics = evaluate_by_interval(X_test, y_test, y_pred)
# print(interval_metrics)

"""y_pred = model.predict_proba(X_test)[:, 1]
print(evaluate_by_interval(X_test, y_test, y_pred))"""

X_train_trans = model.named_steps["standardscaler"].transform(X_train)
logreg        = model.named_steps["logisticregression"]

explainer = shap.LinearExplainer(
    logreg,
    X_train_trans,
)

X_test_trans = model.named_steps["standardscaler"].transform(X_test)
shap_values  = explainer.shap_values(X_test_trans)
shap.summary_plot(shap_values, X_test, feature_names=X_train.columns)
plt.savefig("shap_summary2.png", dpi=300, bbox_inches="tight")
