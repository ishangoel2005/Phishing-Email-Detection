import pandas as pd
import os
import re

def clean_dataset():
    input_path = input("Enter path of input CSV file: ").strip()
    output_path = input("Enter path to save cleaned CSV file (file or folder): ").strip()

    if not os.path.exists(input_path):
        print("❌ Input file does not exist.")
        return

    df = pd.read_csv(input_path)

    required_columns = ['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']
    df = df[required_columns]

    # ---------- DROP ROWS WITH MISSING CRITICAL FIELDS ----------
    critical_cols = ['sender', 'receiver', 'date', 'subject', 'label', 'urls']
    df = df.dropna(subset=critical_cols)

    # ---------- BASIC JUNK FILTER ----------
    junk_keywords = [
        'submission-id',
        'submission id',
        'virus total',
        'submission notes',
        'added:',
        'sender:',
        'receiver:',
    ]

    # ---------- NEWSLETTER / NON-PHISHING FILTER ----------
    newsletter_keywords = [
        'newsletter',
        'edri-gram',
        'biweekly',
        'mailing list',
        'agenda',
        'conference',
        'number ',
        'contents',
        'discuss this story',
        'links:',
        'top issues',
        'posted by',
        'registration at',
        '================================================',
    ]

    MAX_BODY_LENGTH = 4000      # newsletter bodies are HUGE
    MAX_URL_COUNT = 5           # phishing emails usually have few URLs

    def is_bad_row(row):
        combined_text = " ".join(str(v).lower() for v in row)

        # Keyword-based removal
        if any(k in combined_text for k in junk_keywords):
            return True

        if any(k in combined_text for k in newsletter_keywords):
            return True

        # Very long body (newsletter dumps)
        body = str(row['body'])
        if len(body) > MAX_BODY_LENGTH:
            return True

        # Too many URLs
        urls = str(row['urls'])
        url_count = len(re.findall(r'http[s]?://', urls))
        if url_count > MAX_URL_COUNT:
            return True

        return False

    df = df[~df.apply(is_bad_row, axis=1)]

    # ---------- REMOVE EMPTY STRINGS ----------
    for col in critical_cols:
        df = df[df[col].astype(str).str.strip() != ""]

    df = df.reset_index(drop=True)

    # ---------- SAVE ----------
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "CEAS_08_cleaned_final.csv")

    df.to_csv(output_path, index=False)

    print("\n✅ Dataset cleaned successfully!")
    print(f"📊 Final number of rows: {len(df)}")
    print(f"💾 Saved at: {output_path}")


if __name__ == "__main__":
    clean_dataset()
