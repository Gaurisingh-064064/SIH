from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RELATIONSHIPS_PATH = (
    ROOT / "backend" / "data" / "relationships.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(RELATIONSHIPS_PATH)


print("\n================================================")
print("          NYAYANET DATASET INSPECTION")
print("================================================\n")


# ============================================================
# BASIC INFO
# ============================================================

print("Dataset shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n================ CLASS DISTRIBUTION ================\n")

print(
    df["relationship_label"].value_counts()
)

print("\nPercentage distribution:")

print(
    (
        df["relationship_label"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# DUPLICATE PAIR CHECK
# ============================================================

print("\n================ DUPLICATE PAIR CHECK ================\n")


def create_pair_key(row):

    person_a = str(row["person_a_id"])
    person_b = str(row["person_b_id"])

    return "_".join(
        sorted([person_a, person_b])
    )


df["pair_key"] = df.apply(
    create_pair_key,
    axis=1
)


duplicate_count = (
    df["pair_key"]
    .duplicated(keep=False)
    .sum()
)


print(
    "Total relationship records     :",
    len(df)
)

print(
    "Unique person pairs            :",
    df["pair_key"].nunique()
)

print(
    "Records involved in duplicates :",
    duplicate_count
)


if duplicate_count == 0:

    print(
        "\nNo duplicate relationship pairs found."
    )

else:

    print(
        "\nWARNING: Duplicate pairs found!"
    )


# ============================================================
# LABEL CONFLICT CHECK
# ============================================================

print("\n================ LABEL CONFLICT CHECK ================\n")


label_conflicts = (
    df.groupby("pair_key")[
        "relationship_label"
    ]
    .nunique()
)

conflict_count = (
    label_conflicts > 1
).sum()


print(
    "Pairs with conflicting labels :",
    conflict_count
)


if conflict_count == 0:

    print(
        "\nNo conflicting labels found."
    )

else:

    print(
        "\nWARNING: Conflicting labels found!"
    )


# ============================================================
# ACTIVITY COLUMNS
# ============================================================

activity_columns = [

    "phone_call_count",

    "total_call_duration_sec",

    "transaction_count",

    "total_transaction_amount",

    "meeting_count",

    "ground_truth_confidence"
]


# ============================================================
# FEATURE DISTRIBUTION BY CLASS
# ============================================================

print(
    "\n================ FEATURE DISTRIBUTION BY CLASS ================\n"
)


available_columns = [

    column
    for column in activity_columns
    if column in df.columns
]


print(
    df.groupby(
        "relationship_label"
    )[available_columns]
    .mean()
    .round(2)
)


# ============================================================
# MEDIAN DISTRIBUTION
# ============================================================

print(
    "\n================ MEDIAN BY CLASS ================\n"
)


print(
    df.groupby(
        "relationship_label"
    )[available_columns]
    .median()
    .round(2)
)


# ============================================================
# OVERLAP CHECK
# ============================================================

print(
    "\n================ ACTIVITY RANGE BY CLASS ================\n"
)


for column in available_columns:

    print(f"\n{column}")

    summary = df.groupby(
        "relationship_label"
    )[column].agg(
        ["min", "max", "mean", "median"]
    )

    print(
        summary.round(2)
    )


# ============================================================
# RELATIONSHIP TYPE DISTRIBUTION
# ============================================================

print(
    "\n================ RELATIONSHIP TYPE DISTRIBUTION ================\n"
)


if "relationship_type" in df.columns:

    print(
        pd.crosstab(
            df["relationship_type"],
            df["relationship_label"]
        )
    )


# ============================================================
# SAMPLE RECORDS
# ============================================================

print(
    "\n================ SAMPLE POSITIVE RECORDS ================\n"
)


positive_columns = [

    "person_a_name",

    "person_b_name",

    "phone_call_count",

    "transaction_count",

    "meeting_count",

    "relationship_label",

    "ground_truth_confidence"
]


positive_columns = [

    column
    for column in positive_columns
    if column in df.columns
]


print(
    df[
        df["relationship_label"] == 1
    ][positive_columns]
    .sample(
        min(
            5,
            len(
                df[
                    df["relationship_label"] == 1
                ]
            )
        ),
        random_state=42
    )
)


print(
    "\n================ SAMPLE NEGATIVE RECORDS ================\n"
)


print(
    df[
        df["relationship_label"] == 0
    ][positive_columns]
    .sample(
        min(
            5,
            len(
                df[
                    df["relationship_label"] == 0
                ]
            )
        ),
        random_state=42
    )
)


# ============================================================
# REMOVE TEMPORARY COLUMN
# ============================================================

df.drop(
    columns=["pair_key"],
    inplace=True
)


# ============================================================
# FINISHED
# ============================================================

print("\n================================================")
print("        DATASET INSPECTION COMPLETED")
print("================================================\n")