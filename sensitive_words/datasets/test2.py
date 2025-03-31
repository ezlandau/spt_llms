import pandas as pd

# Load the CSV file
df = pd.read_csv("parsed_outputs.csv")

# Define a helper function to count words in the sensitive_word column
def count_words(text):
    # Convert to string and split on whitespace
    return len(str(text).split())

# Add a temporary word count column
df["word_count"] = df["sensitive_word"].apply(count_words)

# Filter rows where sensitive_word has exactly one word
filtered_df = df[df["word_count"] == 1].drop(columns=["word_count"])

# Remove duplicate sensitive words (keeping the first occurrence)
filtered_df = filtered_df.drop_duplicates(subset=["sensitive_word"], keep="first")

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv("filtered_parsed_outputs_unique.csv", index=False)
