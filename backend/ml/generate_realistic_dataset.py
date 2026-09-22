import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

NUM_PERSONS = 600
NUM_RELATIONSHIPS = 660

random.seed(RANDOM_SEED)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "backend" / "data"

PERSONS_PATH = DATA_DIR / "persons.csv"
RELATIONSHIPS_PATH = DATA_DIR / "relationships.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA POOLS
# ============================================================

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Rohan",
    "Rahul", "Amit", "Ravi", "Sahil", "Karan",
    "Vikram", "Aman", "Raj", "Deepak", "Akash",
    "Neha", "Pooja", "Priya", "Anjali", "Kavya",
    "Divya", "Sneha", "Riya", "Shreya", "Nisha",
    "Yash", "Saurabh", "Manish", "Nikhil", "Abhishek"
]

LAST_NAMES = [
    "Sharma", "Verma", "Kumar", "Singh", "Gupta",
    "Khan", "Patel", "Mehta", "Agarwal", "Bansal",
    "Chauhan", "Pandey", "Yadav", "Malhotra", "Khanna"
]

ORGANIZATIONS = [
    "ABC Enterprises",
    "Sharma Trading",
    "Metro Logistics",
    "Kumar Finance",
    "North India Transport",
    "Skyline Services",
    "Global Solutions",
    "Prime Associates",
    "City Construction",
    "Rapid Movers",
    "Independent"
]

LOCATIONS = [
    "Delhi",
    "Noida",
    "Gurugram",
    "Ghaziabad",
    "Faridabad",
    "Meerut",
    "Jaipur",
    "Chandigarh",
    "Lucknow",
    "Agra"
]

RELATIONSHIP_TYPES = [
    "Associate",
    "Colleague",
    "Business Contact",
    "Known Associate",
    "Former Partner",
    "Acquaintance"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def random_date():
    start = datetime(2024, 1, 1)
    end = datetime(2026, 8, 31)

    difference = (end - start).days

    return (
        start +
        timedelta(days=random.randint(0, difference))
    ).strftime("%Y-%m-%d")


def generate_dates(count):
    if count <= 0:
        return ""

    dates = [
        random_date()
        for _ in range(count)
    ]

    return "; ".join(dates)


def generate_amounts(count, strength):
    if count <= 0:
        return "", 0.0

    amounts = []

    for _ in range(count):

        if strength == "high":
            amount = random.randint(
                10000,
                250000
            )

        elif strength == "medium":
            amount = random.randint(
                2000,
                75000
            )

        else:
            amount = random.randint(
                500,
                25000
            )

        amounts.append(amount)

    return (
        "; ".join(map(str, amounts)),
        float(sum(amounts))
    )


def generate_durations(count, strength):
    if count <= 0:
        return "", 0

    durations = []

    for _ in range(count):

        if strength == "high":
            duration = random.randint(
                120,
                3600
            )

        elif strength == "medium":
            duration = random.randint(
                60,
                1800
            )

        else:
            duration = random.randint(
                20,
                900
            )

        durations.append(duration)

    return (
        "; ".join(map(str, durations)),
        sum(durations)
    )


# ============================================================
# GENERATE UNIQUE PERSON NAME
# ============================================================

def generate_unique_name(used_names):
    """
    Generates a unique person name.

    There are fewer base first-name + last-name combinations
    than NUM_PERSONS, so a numeric suffix is added when needed.
    """

    while True:

        base_name = (
            f"{random.choice(FIRST_NAMES)} "
            f"{random.choice(LAST_NAMES)}"
        )

        # First try normal name
        if base_name not in used_names:
            used_names.add(base_name)
            return base_name

        # If already used, create a unique suffix
        suffix = random.randint(1, 99999)

        unique_name = f"{base_name} {suffix}"

        if unique_name not in used_names:
            used_names.add(unique_name)
            return unique_name


# ============================================================
# GENERATE PERSONS
# ============================================================

print("\nGenerating persons...")

persons = []

used_names = set()


for i in range(1, NUM_PERSONS + 1):

    name = generate_unique_name(
        used_names
    )

    person_id = f"P{i:04d}"

    phone_num = (
        "9" +
        "".join(
            str(random.randint(0, 9))
            for _ in range(9)
        )
    )

    vehicle_num = (
        f"DL{random.randint(1, 99):02d}"
        f"AB{random.randint(1000, 9999)}"
    )

    organization = random.choice(
        ORGANIZATIONS
    )

    location = random.choice(
        LOCATIONS
    )

    age = random.randint(
        21,
        65
    )

    persons.append({

        "person_id": person_id,

        "name": name,

        "age": age,

        "phone_num": phone_num,

        "vehicle_num": vehicle_num,

        "org": organization,

        "location": location
    })


persons_df = pd.DataFrame(persons)


# ============================================================
# RELATIONSHIP PAIR TRACKING
# ============================================================

used_pairs = set()


def get_unique_pair():

    while True:

        person_a, person_b = random.sample(
            persons,
            2
        )

        pair = tuple(
            sorted(
                [
                    person_a["person_id"],
                    person_b["person_id"]
                ]
            )
        )

        if pair not in used_pairs:

            used_pairs.add(pair)

            return person_a, person_b


# ============================================================
# GENERATE RELATIONSHIPS
# ============================================================

print("Generating relationships...")


relationships = []


# ============================================================
# RELATIONSHIP CATEGORY DISTRIBUTION
#
# Positive relationships = 220
# Negative relationships = 440
#
# Hard negatives deliberately have significant activity.
# This prevents the model from learning:
#
# "more calls = always relationship"
#
# and makes evaluation more realistic.
# ============================================================

categories = (

    # Strong positive
    ["strong_positive"] * 70 +

    # Weak but genuine relationship
    ["weak_positive"] * 80 +

    # Medium/context-based relationship
    ["context_positive"] * 70 +

    # High activity but NOT meaningful relationship
    ["hard_negative"] * 220 +

    # Low activity negative
    ["weak_negative"] * 220
)


random.shuffle(categories)


# ============================================================
# CREATE RELATIONSHIP RECORDS
# ============================================================

for index, category in enumerate(
    categories,
    start=1
):

    person_a, person_b = get_unique_pair()


    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    calls = 0

    transactions = 0

    meetings = 0

    activity_strength = "low"

    label = 0

    relationship_type = random.choice(
        RELATIONSHIP_TYPES
    )


    # ========================================================
    # STRONG POSITIVE
    #
    # Multiple strong evidence sources
    # ========================================================

    if category == "strong_positive":

        calls = random.randint(
            12,
            45
        )

        transactions = random.randint(
            3,
            10
        )

        meetings = random.randint(
            2,
            8
        )

        activity_strength = "high"

        label = 1


    # ========================================================
    # WEAK POSITIVE
    #
    # Genuine relationship with limited evidence
    # ========================================================

    elif category == "weak_positive":

        calls = random.randint(
            1,
            10
        )

        transactions = random.randint(
            0,
            2
        )

        meetings = random.randint(
            0,
            3
        )

        activity_strength = "low"

        label = 1


    # ========================================================
    # CONTEXT POSITIVE
    #
    # Moderate activity
    # ========================================================

    elif category == "context_positive":

        calls = random.randint(
            4,
            20
        )

        transactions = random.randint(
            0,
            5
        )

        meetings = random.randint(
            1,
            5
        )

        activity_strength = "medium"

        label = 1


    # ========================================================
    # HARD NEGATIVE
    #
    # High interaction does NOT necessarily mean
    # meaningful suspicious relationship.
    #
    # These make the model harder and more realistic.
    # ========================================================

    elif category == "hard_negative":

        calls = random.randint(
            5,
            30
        )

        transactions = random.randint(
            1,
            8
        )

        meetings = random.randint(
            1,
            6
        )

        activity_strength = random.choice(
            [
                "medium",
                "high"
            ]
        )

        label = 0

        relationship_type = random.choice(
            [
                "Business Contact",
                "Colleague",
                "Acquaintance"
            ]
        )


    # ========================================================
    # WEAK NEGATIVE
    # ========================================================

    elif category == "weak_negative":

        calls = random.randint(
            0,
            6
        )

        transactions = random.randint(
            0,
            2
        )

        meetings = random.randint(
            0,
            2
        )

        activity_strength = "low"

        label = 0


    # ========================================================
    # PHONE CALL DATA
    # ========================================================

    call_dates = generate_dates(
        calls
    )

    (
        call_durations,
        total_duration
    ) = generate_durations(
        calls,
        activity_strength
    )


    # ========================================================
    # TRANSACTION DATA
    # ========================================================

    transaction_dates = generate_dates(
        transactions
    )

    (
        transaction_amounts,
        total_amount
    ) = generate_amounts(
        transactions,
        activity_strength
    )


    # ========================================================
    # MEETING DATA
    # ========================================================

    meeting_dates = generate_dates(
        meetings
    )

    meeting_locations = ""


    if meetings > 0:

        meeting_location_list = []

        for _ in range(meetings):

            meeting_location_list.append(
                random.choice(
                    LOCATIONS
                )
            )

        meeting_locations = "; ".join(
            meeting_location_list
        )


    # ========================================================
    # GROUND TRUTH CONFIDENCE
    # ========================================================

    if label == 1:

        if category == "strong_positive":

            confidence = random.uniform(
                0.75,
                0.98
            )

        elif category == "context_positive":

            confidence = random.uniform(
                0.55,
                0.88
            )

        else:

            confidence = random.uniform(
                0.45,
                0.75
            )

    else:

        if category == "hard_negative":

            confidence = random.uniform(
                0.20,
                0.55
            )

        else:

            confidence = random.uniform(
                0.01,
                0.35
            )


    # ========================================================
    # RELATIONSHIP DESCRIPTION
    # ========================================================

    if label == 1:

        description = (
            f"Relationship evidence between "
            f"{person_a['name']} and "
            f"{person_b['name']}. "
            f"Phone records show {calls} calls. "
            f"Transaction records show "
            f"{transactions} transactions. "
            f"Meeting records show {meetings} meetings."
        )

    else:

        description = (
            f"Observed interaction between "
            f"{person_a['name']} and "
            f"{person_b['name']}. "
            f"Available records include "
            f"{calls} calls, "
            f"{transactions} transactions, "
            f"and {meetings} meetings. "
            f"Evidence does not establish a "
            f"meaningful relationship."
        )


    # ========================================================
    # CREATE RELATIONSHIP ROW
    # ========================================================

    relationships.append({

        "relationship_id":
            f"REL{index:05d}",

        "person_a_id":
            person_a["person_id"],

        "person_a_name":
            person_a["name"],

        "person_b_id":
            person_b["person_id"],

        "person_b_name":
            person_b["name"],

        "phone_call_count":
            calls,

        "phone_call_dates":
            call_dates,

        "phone_call_durations_sec":
            call_durations,

        "total_call_duration_sec":
            total_duration,

        "transaction_count":
            transactions,

        "transaction_dates":
            transaction_dates,

        "transaction_amounts":
            transaction_amounts,

        "total_transaction_amount":
            total_amount,

        "meeting_count":
            meetings,

        "meeting_dates":
            meeting_dates,

        "meeting_locations":
            meeting_locations,

        "relationship_label":
            label,

        "ground_truth_confidence":
            round(
                confidence,
                4
            ),

        "relationship_type":
            relationship_type,

        "relationship_description":
            description
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

relationships_df = pd.DataFrame(
    relationships
)


# ============================================================
# SAVE DATA
# ============================================================

persons_df.to_csv(
    PERSONS_PATH,
    index=False
)

relationships_df.to_csv(
    RELATIONSHIPS_PATH,
    index=False
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n================ DATASET GENERATED ================\n")

print(
    f"Persons generated: "
    f"{len(persons_df)}"
)

print(
    f"Relationships generated: "
    f"{len(relationships_df)}"
)


print(
    "\nRelationship class distribution:\n"
)

print(
    relationships_df[
        "relationship_label"
    ].value_counts()
)


print("\nPositive relationships:")

print(
    len(
        relationships_df[
            relationships_df[
                "relationship_label"
            ] == 1
        ]
    )
)


print("\nNegative relationships:")

print(
    len(
        relationships_df[
            relationships_df[
                "relationship_label"
            ] == 0
        ]
    )
)


print("\nFiles saved:\n")

print(
    f"Persons: "
    f"{PERSONS_PATH}"
)

print(
    f"Relationships: "
    f"{RELATIONSHIPS_PATH}"
)


print(
    "\nDataset generation completed successfully.\n"
)