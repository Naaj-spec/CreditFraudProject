import pandas as pd

# Load original dataset (from Kaggle)
df = pd.read_csv("data/creditcard.csv")

# Take a small sample for testing
new_data = df.sample(20).drop(columns=["Class"])  # remove labels

# Save it
new_data.to_csv("data/new_transactions.csv", index=False)
print(" Test CSV created")
