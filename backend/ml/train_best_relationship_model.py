import math
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

RELATIONSHIPS_PATH = ROOT / "backend" / "data" / "relationships.csv"
PERSONS_PATH = ROOT / "backend" / "data" / "persons.csv"

MODEL_PATH = ROOT / "ml" / "relationship_model.joblib"

METADATA_PATH = ROOT / "ml" / "relationship_model_metadata.joblib"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

relationships = pd.read_csv(RELATIONSHIPS_PATH)
persons = pd.read_csv(PERSONS_PATH)

print(f"Relationships loaded: {relationships.shape}")
print(f"Persons loaded      : {persons.shape}")


# ============================================================
# NORMALIZE PERSON DATA
# ============================================================

persons["person_id"] = persons["person_id"].astype(str)

person_lookup = persons.set_index("person_id").to_dict("index")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(value):

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def get_person(person_id):

    return person_lookup.get(str(person_id), {})


def shared_value(person_a, person_b, field):

    a = normalize(person_a.get(field))
    b = normalize(person_b.get(field))

    if a and b and a == b:
        return 1

    return 0


# ============================================================
# CREATE RELATIONSHIP FEATURES
# ============================================================

print("\nCreating relationship features...")

feature_rows = []

for _, row in relationships.iterrows():

    person_a = get_person(row["person_a_id"])
    person_b = get_person(row["person_b_id"])

    # --------------------------------------------------------
    # RAW ACTIVITY FEATURES
    # --------------------------------------------------------

    calls = float(row.get("phone_call_count", 0) or 0)

    duration = float(
        row.get("total_call_duration_sec", 0) or 0
    )

    transactions = float(
        row.get("transaction_count", 0) or 0
    )

    amount = float(
        row.get("total_transaction_amount", 0) or 0
    )

    meetings = float(
        row.get("meeting_count", 0) or 0
    )


    # --------------------------------------------------------
    # SOURCE DIVERSITY
    # --------------------------------------------------------

    source_diversity = sum([

        1 if calls > 0 else 0,

        1 if transactions > 0 else 0,

        1 if meetings > 0 else 0,

    ])


    # --------------------------------------------------------
    # FINAL FEATURE ROW
    # --------------------------------------------------------

    feature_rows.append([

        math.log1p(calls),

        math.log1p(duration),

        math.log1p(transactions),

        math.log1p(amount),

        math.log1p(meetings),

        source_diversity,

    ])


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
# CREATE FEATURE MATRIX
# ============================================================

X = pd.DataFrame(

    feature_rows,

    columns=feature_names

)


# ============================================================
# TARGET
# ============================================================

y = pd.to_numeric(

    relationships["relationship_label"],

    errors="coerce"

)


# Remove invalid labels

valid = y.notna()

X = X.loc[valid].reset_index(drop=True)

y = y.loc[valid].astype(int).reset_index(drop=True)


# ============================================================
# DATA INFORMATION
# ============================================================

print("\n================ DATA INFO ================\n")

print("Total rows:", len(X))

print("\nClass distribution:")

print(y.value_counts())

print("\nFeatures used:")

for feature in feature_names:

    print(" -", feature)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y,

)


print("\nTrain rows:", len(X_train))

print("Test rows :", len(X_test))


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest model...")


model = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1,

)


model.fit(

    X_train,

    y_train

)


print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


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


auc = roc_auc_score(

    y_test,

    y_prob

)


# ============================================================
# RESULTS
# ============================================================

print("\n")

print("=" * 60)

print("       NYAYANET RELATIONSHIP MODEL")

print("       RANDOM FOREST EVALUATION")

print("=" * 60)


print(f"\nAccuracy : {accuracy * 100:.2f}%")

print(f"Precision: {precision * 100:.2f}%")

print(f"Recall   : {recall * 100:.2f}%")

print(f"F1 Score : {f1 * 100:.2f}%")

print(f"ROC-AUC  : {auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(

    confusion_matrix(

        y_test,

        y_pred

    )

)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(

    classification_report(

        y_test,

        y_pred,

        zero_division=0

    )

)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n================ FEATURE IMPORTANCE ================\n")

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": model.feature_importances_

})


importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)


for _, row in importance_df.iterrows():

    print(

        f"{row['Feature']:<30} "

        f"{row['Importance']:.4f}"

    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    model,

    MODEL_PATH

)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model_type": "RandomForestClassifier",

    "feature_names": feature_names,

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "roc_auc": float(auc),

}


joblib.dump(

    metadata,

    METADATA_PATH

)


# ============================================================
# SUCCESS
# ============================================================

print("\n")

print("=" * 60)

print("MODEL SAVED SUCCESSFULLY")

print("=" * 60)


print("\nModel path:")

print(MODEL_PATH)


print("\nMetadata path:")

print(METADATA_PATH)


print("\nFinal model type:")

print("Random Forest")


print("\nTraining finished successfully.\n")