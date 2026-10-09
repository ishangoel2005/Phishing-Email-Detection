import pandas as pd
import os

def combine_csv_files():
    print("Enter paths of CSV files to combine (one per line).")
    print("After entering all paths, press ENTER on an empty line.\n")

    file_paths = []
    while True:
        path = input("CSV file path: ").strip()
        if path == "":
            break
        if not os.path.exists(path):
            print("❌ File does not exist. Try again.")
            continue
        file_paths.append(path)

    if len(file_paths) < 2:
        print("❌ Please provide at least two CSV files.")
        return

    output_path = input("\nEnter output CSV file path (including filename): ").strip()

    required_columns = ['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']
    dataframes = []

    for file in file_paths:
        df = pd.read_csv(file)

        # Ensure same schema
        if list(df.columns) != required_columns:
            print(f"❌ Column mismatch in file: {file}")
            print(f"Expected: {required_columns}")
            print(f"Found: {list(df.columns)}")
            return

        dataframes.append(df)

    # Combine all CSVs
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save combined CSV
    combined_df.to_csv(output_path, index=False)

    print("\n✅ CSV files combined successfully!")
    print(f"📄 Total rows: {len(combined_df)}")
    print(f"💾 Saved at: {output_path}")

if __name__ == "__main__":
    combine_csv_files()