import pandas as pd

# -----------------------------
# File paths (EDIT THESE)
# -----------------------------
OLD_DATASET_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/archive/CLEANED/combined.csv"
NEW_DATASET_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/phishing_dataset_40000.csv"
OUTPUT_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/custom_combined_dataset.csv"

# -----------------------------
# Load datasets
# -----------------------------
old_df = pd.read_csv(OLD_DATASET_PATH, encoding="latin1")
new_df = pd.read_csv(NEW_DATASET_PATH, encoding="latin1")

print("Old dataset shape:", old_df.shape)
print("New dataset shape:", new_df.shape)

# -----------------------------
# Clean OLD dataset
# Remove: receiver, date, urls
# Keep: sender, subject, body, label
# -----------------------------
old_df_clean = old_df.drop(columns=["receiver", "date", "urls"], errors="ignore")
old_df_clean = old_df_clean[["sender", "subject", "body", "label"]]

# -----------------------------
# Clean NEW dataset
# Remove: email_id
# Keep: sender, subject, body, label
# -----------------------------
new_df_clean = new_df.drop(columns=["email_id"], errors="ignore")
new_df_clean = new_df_clean[["sender", "subject", "body", "label"]]

# -----------------------------
# Combine datasets
# -----------------------------
combined_df = pd.concat([old_df_clean, new_df_clean], ignore_index=True)

# Shuffle the dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# Save final dataset
# -----------------------------
combined_df.to_csv(OUTPUT_PATH, index=False)

print("Final combined dataset saved to:", OUTPUT_PATH)
print("Final dataset shape:", combined_df.shape)
print("Label distribution:")
print(combined_df["label"].value_counts())