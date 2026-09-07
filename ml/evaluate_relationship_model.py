import math
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
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


# ============================================================
# LOAD DATA
# ============================================================

relationships = pd.read_csv(RELATIONSHIPS_PATH)
persons = pd.read_csv(PERSONS_PATH)

print("\n================ DATASET INFO ================\n")
print("Relationships shape:", relationships.shape)
print("Persons shape      :", persons.shape)

print("\nRelationship columns:")
print(relationships.columns.tolist())


# ============================================================
# NORMALIZE PERSON DATA
# ============================================================

persons["person_id"] = persons["person_id"].astype(str)

person_lookup = persons.set_index("person_id").to_dict("index")


# ============================================================
# HELPER
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
# BUILD THE SAME FEATURES USED BY NYAYANET
# ============================================================

feature_rows = []

for _, row in relationships.iterrows():

    person_a = get_person(row["person_a_id"])
    person_b = get_person(row["person_b_id"])

    # Raw relationship activity
    calls = float(row.get("phone_call_count", 0) or 0)
    duration = float(row.get("total_call_duration_sec", 0) or 0)
    transactions = float(row.get("transaction_count", 0) or 0)
    amount = float(row.get("total_transaction_amount", 0) or 0)
    meetings = float(row.get("meeting_count", 0) or 0)

    # This CSV represents one relationship record.
    # The live backend increments co_occurrences for every
    # evidence source record. For this training CSV, one row
    # corresponds to one relationship record.
    co_occurrences = 1

    # Number of observable activity types
    source_diversity = sum(
        [
            1 if calls > 0 else 0,
            1 if transactions > 0 else 0,
            1 if meetings > 0 else 0,
        ]
    )

    # Shared identifiers/context
    shared_phone = shared_value(person_a, person_b, "phone_num")

    shared_vehicle = shared_value(person_a, person_b, "vehicle_num")

    shared_org = shared_value(person_a, person_b, "org")

    shared_location = shared_value(person_a, person_b, "location")

    # IMPORTANT:
    # Same transformations as backend relationship_features()
    feature_rows.append(
        [
            math.log1p(calls),
            math.log1p(duration),
            math.log1p(transactions),
            math.log1p(amount),
            math.log1p(meetings),
            math.log1p(co_occurrences),
            source_diversity,
            shared_phone,
            shared_vehicle,
            shared_org,
            shared_location,
        ]
    )


# ============================================================
# CREATE FEATURE MATRIX
# ============================================================

feature_names = [
    "log_calls",
    "log_call_duration",
    "log_transactions",
    "log_transaction_amount",
    "log_meetings",
    "log_co_occurrences",
    "source_diversity",
    "shared_phone",
    "shared_vehicle",
    "shared_org",
    "shared_location",
]

X = pd.DataFrame(feature_rows, columns=feature_names)


# ============================================================
# TARGET
# ============================================================

print("\nTarget column: relationship_label")

y = pd.to_numeric(relationships["relationship_label"], errors="coerce")

# Remove rows with invalid labels
valid = y.notna()

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].astype(int).reset_index(drop=True)

print("\nFinal rows:", len(X))

print("\nClass distribution:")
print(y.value_counts())


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
# LOAD SAVED MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("\nLoaded model:")
print(model)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

try:
    y_prob = model.predict_proba(X_test)[:, 1]
except Exception:
    y_prob = None


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, zero_division=0)

recall = recall_score(y_test, y_pred, zero_division=0)

f1 = f1_score(y_test, y_pred, zero_division=0)


print("\n")
print("=" * 60)
print("       NYAYANET RELATIONSHIP MODEL")
print("              EVALUATION")
print("=" * 60)

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")


# ============================================================
# ROC-AUC
# ============================================================

if y_prob is not None:

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC  : {auc:.4f}")
    except ValueError:
        print("ROC-AUC  : unavailable")

else:
    print("ROC-AUC  : unavailable")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nEvaluation finished successfully.")
