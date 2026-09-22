import math
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "backend" / "data" / "relationships.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_number(value):

    try:
        if pd.isna(value):
            return 0.0

        return float(value)

    except Exception:
        return 0.0


# ============================================================
# CREATE FEATURES
# ============================================================

rows = []

for _, row in df.iterrows():

    calls = safe_number(
        row.get("phone_call_count", 0)
    )

    duration = safe_number(
        row.get("total_call_duration_sec", 0)
    )

    transactions = safe_number(
        row.get("transaction_count", 0)
    )

    amount = safe_number(
        row.get("total_transaction_amount", 0)
    )

    meetings = safe_number(
        row.get("meeting_count", 0)
    )

    source_diversity = sum([
        calls > 0,
        transactions > 0,
        meetings > 0
    ])

    rows.append([

        math.log1p(calls),

        math.log1p(duration),

        math.log1p(transactions),

        math.log1p(amount),

        math.log1p(meetings),

        source_diversity,

    ])


# ============================================================
# FEATURE MATRIX
# ============================================================

feature_names = [

    "log_calls",

    "log_call_duration",

    "log_transactions",

    "log_transaction_amount",

    "log_meetings",

    "source_diversity",

]


X = pd.DataFrame(
    rows,
    columns=feature_names
)


y = pd.to_numeric(
    df["relationship_label"],
    errors="coerce"
)


valid = y.notna()

X = X.loc[valid].reset_index(drop=True)

y = y.loc[valid].astype(int).reset_index(drop=True)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":

        Pipeline([

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",

                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42
                )
            )

        ]),


    "Random Forest":

        RandomForestClassifier(

            n_estimators=300,

            max_depth=8,

            min_samples_split=8,

            min_samples_leaf=4,

            class_weight="balanced",

            random_state=42

        ),


    "Gradient Boosting":

        GradientBoostingClassifier(

            n_estimators=100,

            learning_rate=0.05,

            max_depth=3,

            random_state=42

        )

}


# ============================================================
# RESULTS
# ============================================================

print("\n")

print("=" * 85)

print(
    "NYAYANET RELATIONSHIP MODEL COMPARISON"
)

print("=" * 85)


results = []


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )


    model.fit(
        X_train,
        y_train
    )


    y_pred = model.predict(
        X_test
    )


    y_prob = model.predict_proba(
        X_test
    )[:, 1]


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    auc = roc_auc_score(
        y_test,
        y_prob
    )


    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": auc

    })


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


for column in [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score",

    "ROC-AUC"

]:

    results_df[column] = (
        results_df[column] * 100
    ).round(2)


print("\n")

print(results_df.to_string(
    index=False
))


print("\n")

print("=" * 85)

print(
    "COMPARISON FINISHED"
)

print("=" * 85)