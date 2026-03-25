"""Generate 1M synthetic messy records for performance testing."""

import random
import string
import csv
from pathlib import Path

FIRST_NAMES = [
    "John", "Jane", "Robert", "Mary", "James", "Patricia", "Michael", "Jennifer",
    "William", "Linda", "David", "Elizabeth", "Richard", "Barbara", "Joseph", "Susan",
    "Thomas", "Jessica", "Charles", "Sarah", "Christopher", "Karen", "Daniel", "Nancy",
    "Matthew", "Lisa", "Anthony", "Betty", "Mark", "Margaret", "Donald", "Sandra",
    "Steven", "Ashley", "Paul", "Dorothy", "Andrew", "Kimberly", "Joshua", "Emily",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen",
    "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera",
]

STATES = ["PA", "NJ", "NY", "DE", "MD", "CT", "VA", "OH", "CA", "TX", "FL", "IL"]

SPECIALTIES = [
    "Cardiology", "Oncology", "Neurology", "Orthopedics", "Dermatology",
    "Pediatrics", "Radiology", "Psychiatry", "Urology", "Gastroenterology",
    "Endocrinology", "Pulmonology", "Nephrology", "Rheumatology", "Hematology",
]

DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "aol.com", "hospital.org", "clinic.com", "health.net"]

STREETS = ["Main St", "Oak Ave", "Elm Blvd", "Pine Dr", "Maple Ln", "Cedar Rd", "Birch Ct", "Walnut Way"]
CITIES = ["Philadelphia", "New York", "Newark", "Wilmington", "Baltimore", "Hartford", "Richmond"]


def random_phone():
    return f"{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"


def random_zip():
    return f"{random.randint(10000, 99999)}"


def random_address():
    return f"{random.randint(1, 9999)} {random.choice(STREETS)}"


def mess_up_value(value, mess_type):
    """Apply random messiness to a value."""
    if value is None:
        return value
    if mess_type == "case":
        return random.choice([value.upper(), value.lower(), value.title(), value])
    elif mess_type == "whitespace":
        spaces = " " * random.randint(1, 4)
        return random.choice([spaces + value, value + spaces, spaces + value + spaces])
    elif mess_type == "typo":
        if len(value) > 2:
            i = random.randint(1, len(value) - 2)
            return value[:i] + random.choice(string.ascii_lowercase) + value[i+1:]
        return value
    elif mess_type == "null":
        return random.choice(["", None, "NULL", "N/A", "  "])
    elif mess_type == "phone_format":
        digits = value.replace("-", "")
        formats = [
            digits,
            f"({digits[:3]}) {digits[3:6]}-{digits[6:]}",
            f"{digits[:3]}.{digits[3:6]}.{digits[6:]}",
            f"1-{value}",
            f"+1{digits}",
        ]
        return random.choice(formats)
    elif mess_type == "email_mess":
        return random.choice([value.upper(), " " + value, value + " ", value.replace("@", " @ ")])
    return value


def generate(output_path: Path, n_records: int = 1_000_000, dupe_rate: float = 0.15):
    """Generate synthetic data with controlled duplicates and messiness.

    Args:
        output_path: Where to write the CSV.
        n_records: Total records to generate.
        dupe_rate: Fraction of records that are duplicates of existing ones.
    """
    random.seed(42)

    # Generate base records (non-duplicates)
    n_unique = int(n_records * (1 - dupe_rate))
    n_dupes = n_records - n_unique

    print(f"Generating {n_unique:,} unique records + {n_dupes:,} duplicates = {n_records:,} total")

    base_records = []
    for i in range(n_unique):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        email = f"{first.lower()}.{last.lower()}{random.randint(1,999)}@{random.choice(DOMAINS)}"

        record = {
            "id": i + 1,
            "first_name": first,
            "last_name": last,
            "email": email,
            "phone": random_phone(),
            "address": random_address(),
            "city": random.choice(CITIES),
            "state": random.choice(STATES),
            "zip": random_zip(),
            "specialty": random.choice(SPECIALTIES),
        }
        base_records.append(record)

    # Generate duplicates with messiness
    all_records = list(base_records)

    for _ in range(n_dupes):
        original = random.choice(base_records)
        dupe = dict(original)
        dupe["id"] = len(all_records) + 1

        # Apply 1-3 types of messiness
        n_messes = random.randint(1, 3)
        mess_options = [
            ("first_name", "case"),
            ("first_name", "typo"),
            ("first_name", "whitespace"),
            ("last_name", "case"),
            ("last_name", "typo"),
            ("last_name", "whitespace"),
            ("email", "email_mess"),
            ("email", "case"),
            ("phone", "phone_format"),
            ("address", "case"),
            ("address", "whitespace"),
            ("city", "case"),
            ("state", "case"),
            ("zip", "whitespace"),
            ("specialty", "case"),
            ("specialty", "typo"),
        ]

        # Randomly null out 0-2 fields
        if random.random() < 0.3:
            null_fields = random.sample(
                ["phone", "email", "address", "specialty"],
                k=random.randint(1, 2)
            )
            for f in null_fields:
                dupe[f] = mess_up_value(dupe[f], "null")

        for field, mess_type in random.sample(mess_options, min(n_messes, len(mess_options))):
            if dupe[field] is not None and dupe[field] not in ("", "NULL", "N/A"):
                dupe[field] = mess_up_value(dupe[field], mess_type)

        all_records.append(dupe)

    # Shuffle
    random.shuffle(all_records)

    # Add some completely junk rows (~0.1%)
    n_junk = int(n_records * 0.001)
    for _ in range(n_junk):
        idx = random.randint(0, len(all_records) - 1)
        junk_type = random.choice(["empty", "garbage", "null_row"])
        if junk_type == "empty":
            all_records[idx] = {k: "" for k in all_records[idx]}
        elif junk_type == "garbage":
            all_records[idx] = {k: "".join(random.choices(string.printable[:62], k=random.randint(1, 20))) for k in all_records[idx]}
        else:
            all_records[idx] = {k: None for k in all_records[idx]}

    # Write CSV
    print(f"Writing to {output_path}...")
    fieldnames = ["id", "first_name", "last_name", "email", "phone", "address", "city", "state", "zip", "specialty"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_records:
            # Convert None to empty string for CSV
            row = {k: (v if v is not None else "") for k, v in record.items()}
            writer.writerow(row)

    print(f"Done. File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Unique base records: {n_unique:,}")
    print(f"  Duplicates (messy): {n_dupes:,}")
    print(f"  Junk rows: {n_junk:,}")


if __name__ == "__main__":
    output = Path("D:/show_case/goldenmatch/tests/fixtures/synthetic_1m.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    generate(output)
