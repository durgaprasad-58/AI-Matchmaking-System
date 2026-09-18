import pandas as pd
import random
from datetime import datetime, timedelta

feedback_data = []

# Date range
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)

# Generate feedback for 80 users
for i in range(1, 81):

    user_id = f"X{i:02d}"

    # Each user gives feedback to 5 other users
    for j in range(5):

        matched_user = f"X{random.randint(1, 80):02d}"

        # Avoid self-matching
        while matched_user == user_id:
            matched_user = f"X{random.randint(1, 80):02d}"

        action = random.choice([0, 1])

        # Generate random date
        random_days = random.randint(
            0,
            (end_date - start_date).days
        )

        random_date = (
            start_date + timedelta(days=random_days)
        ).strftime("%Y-%m-%d")

        feedback_data.append([
            user_id,
            matched_user,
            action,
            random_date
        ])

# Create DataFrame
feedback_df = pd.DataFrame(
    feedback_data,
    columns=[
        "user_id",
        "matched_user_id",
        "action",
        "timestamp"
    ]
)

# Save CSV
feedback_df.to_csv("feedback.csv", index=False)
print(feedback_df.info())
print(feedback_df)
