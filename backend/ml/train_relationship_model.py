import math
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RELATIONSHIPS_PATH = (
    ROOT / "backend" / "data" / "relationships.csv"
)

MODEL_PATH = (
    ROOT / "ml" / "relationship_model.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    RELATIONSHIPS_PATH
)

print(
    f"Dataset loaded: {df.shape}"
)


# ============================================================
# CREATE FEATURES
# ============================================================

print(
    "\nCreating relationship features..."
)


def safe_number(value):

    try:

        if pd.isna(value):

            return 0.0

        return float(value)

    except Exception:

        return 0.0


feature_rows = []


for _, row in df.iterrows():

    # --------------------------------------------------------
    # RAW ACTIVITY
    # --------------------------------------------------------

    calls = safe_number(
        row.get(
            "phone_call_count",
            0
        )
    )

    duration = safe_number(
        row.get(
            "total_call_duration_sec",
            0
        )
    )

    transactions = safe_number(
        row.get(
            "transaction_count",
            0
        )
    )

    amount = safe_number(
        row.get(
            "total_transaction_amount",
            0
        )
    )

    meetings = safe_number(
        row.get(
            "meeting_count",
            0
        )
    )


    # --------------------------------------------------------
    # SOURCE DIVERSITY
    # --------------------------------------------------------

    source_diversity = sum(

        [

            1 if calls > 0 else 0,

            1 if transactions > 0 else 0,

            1 if meetings > 0 else 0,

        ]

    )


    # --------------------------------------------------------
    # TRANSFORM FEATURES
    # --------------------------------------------------------

    feature_rows.append(

        [

            math.log1p(calls),

            math.log1p(duration),

            math.log1p(transactions),

            math.log1p(amount),

            math.log1p(meetings),

            source_diversity,

        ]

    )


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [

    "log_calls",

    "log_call_duration",

    "log_transactions",

    "log_transaction_amount",

    "log_meetings",

    "source_diversity",

]


# ============================================================
# FEATURE MATRIX
# ============================================================

X = pd.DataFrame(

    feature_rows,

    columns=feature_names

)


# ============================================================
# TARGET
# ============================================================

y = pd.to_numeric(

    df[
        "relationship_label"
    ],

    errors="coerce"

)


valid_rows = y.notna()


X = X.loc[
    valid_rows
].reset_index(
    drop=True
)


y = y.loc[
    valid_rows
].astype(
    int
).reset_index(
    drop=True
)


# ============================================================
# DATA INFO
# ============================================================

print("\n================ DATA INFO ================\n")

print(
    "Total rows:",
    len(X)
)


print(
    "\nClass distribution:"
)


print(
    y.value_counts()
)


print(
    "\nFeatures used:"
)


for feature in feature_names:

    print(
        " -",
        feature
    )


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "\nTrain rows:",
    len(X_train)
)


print(
    "Test rows:",
    len(X_test)
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nTraining Logistic Regression model..."
)


model = Pipeline(

    [

        (

            "scaler",

            StandardScaler()

        ),

        (

            "classifier",

            LogisticRegression(

                class_weight="balanced",

                max_iter=3000,

                random_state=42

            )

        )

    ]

)


# ============================================================
# TRAIN
# ============================================================

model.fit(

    X_train,

    y_train

)


print(
    "Training completed."
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(

    X_test

)


y_prob = model.predict_proba(

    X_test

)[:, 1]


# ============================================================
# METRICS
# ============================================================

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


try:

    auc = roc_auc_score(

        y_test,

        y_prob

    )

except Exception:

    auc = None


# ============================================================
# RESULTS
# ============================================================

print("\n")

print("=" * 60)

print(
    "       NYAYANET RELATIONSHIP MODEL"
)

print(
    "          TRAINING RESULTS"
)

print("=" * 60)


print(
    f"\nAccuracy : {accuracy * 100:.2f}%"
)


print(
    f"Precision: {precision * 100:.2f}%"
)


print(
    f"Recall   : {recall * 100:.2f}%"
)


print(
    f"F1 Score : {f1 * 100:.2f}%"
)


if auc is not None:

    print(
        f"ROC-AUC  : {auc:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:"
)


print(

    confusion_matrix(

        y_test,

        y_pred

    )

)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:\n"
)


print(

    classification_report(

        y_test,

        y_pred,

        zero_division=0

    )

)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    model,

    MODEL_PATH

)


print(
    "\n================================================"
)


print(
    "MODEL SAVED SUCCESSFULLY"
)


print(
    "================================================"
)


print(
    f"\nModel path:\n{MODEL_PATH}"
)


# ============================================================
# SAVE FEATURE METADATA
# ============================================================

metadata = {

    "feature_names": feature_names,

    "model_type": "LogisticRegression",

    "dataset_rows": len(X),

}


METADATA_PATH = (

    ROOT
    / "ml"
    / "relationship_model_metadata.joblib"

)


joblib.dump(

    metadata,

    METADATA_PATH

)


print(
    f"\nMetadata path:\n{METADATA_PATH}"
)


print(
    "\nTraining finished successfully.\n"
)