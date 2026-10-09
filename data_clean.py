import pandas as pd
import os

def clean_dataset():
    # ---------- USER INPUT ----------
    input_path = input("Enter path of input CSV file: ").strip()
    output_path = input("Enter path to save cleaned CSV file: ").strip()

    if not os.path.exists(input_path):
        print("❌ Input file does not exist.")
        return

    # ---------- LOAD DATA ----------
    df = pd.read_csv(input_path)

    required_columns = ['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']

    # ---------- KEEP ONLY REQUIRED COLUMNS ----------
    df = df[required_columns]

    # ---------- REMOVE ROWS WITH MISSING CRITICAL FIELDS ----------
    # body is allowed to be empty
    critical_cols = ['sender', 'receiver', 'date', 'subject', 'label', 'urls']
    df = df.dropna(subset=critical_cols)

    # ---------- REMOVE JUNK / METADATA ROWS ----------
    junk_keywords = [
        'submission-id',
        'submission id',
        'virus total',
        'submission notes',
        'added:',
        'sender:',
        'receiver:',
    ]

    def is_junk_row(row):
        combined_text = " ".join(str(value).lower() for value in row)
        return any(keyword in combined_text for keyword in junk_keywords)

    df = df[~df.apply(is_junk_row, axis=1)]

    # ---------- REMOVE EMPTY STRINGS ----------
    for col in critical_cols:
        df = df[df[col].astype(str).str.strip() != ""]

    # ---------- RESET INDEX ----------
    df = df.reset_index(drop=True)

    # ---------- SAVE CLEAN DATA ----------
    df.to_csv(output_path, index=False)

    print("\n✅ Dataset cleaned successfully!")
    print(f"📊 Final number of rows: {len(df)}")
    print(f"💾 Saved at: {output_path}")


# ---------- RUN ----------
if __name__ == "__main__":
    clean_dataset()