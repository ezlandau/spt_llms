import pandas as pd
import re


def correct_prefix(text, sensitive_word):
    text = str(text)
    sensitive_lower = sensitive_word.lower()
    tokens = re.findall(r'\w+|\W+', text)

    # Remove all occurrences of the sensitive word (case-insensitive)
    filtered_tokens = [
        token for token in tokens
        if not (token.strip() and re.fullmatch(r'\w+', token) and token.lower() == sensitive_lower)
    ]

    # Capitalize the sensitive word and place at the start
    capitalized = sensitive_word[0].upper() + sensitive_word[1:]
    if filtered_tokens and re.match(r'^\W+$', filtered_tokens[0]):
        # Insert after leading punctuation
        filtered_tokens.insert(0, capitalized)
    else:
        filtered_tokens = [capitalized] + ([' '] if filtered_tokens else []) + filtered_tokens

    # Rebuild text and clean spaces
    corrected = ''.join(filtered_tokens)
    corrected = re.sub(r'\s+', ' ', corrected).strip()
    return corrected


def correct_suffix(text, sensitive_word):
    text = str(text)
    sensitive_lower = sensitive_word.lower()
    tokens = re.findall(r'\w+|\W+', text)
    trailing_punct = []

    # Extract trailing punctuation (non-word tokens at the end)
    while tokens and not re.fullmatch(r'\w+', tokens[-1]):
        trailing_punct.insert(0, tokens.pop())

    # Remove all sensitive word occurrences (case-insensitive)
    filtered_tokens = [
        token for token in tokens
        if not (token.strip() and re.fullmatch(r'\w+', token) and token.lower() == sensitive_lower)
    ]

    # Add sensitive word before trailing punctuation
    filtered_tokens.append(sensitive_word)
    corrected_tokens = filtered_tokens + trailing_punct

    # Rebuild text and clean spaces
    corrected = ''.join(corrected_tokens)
    corrected = re.sub(r'\s+', ' ', corrected).strip()
    return corrected


# Load and process data
df = pd.read_csv("NEW_outputs_safetyeval.csv")

condition = (df["Checked"] != "YES") & ((df["is_benign_safe"] == "No") | (df["is_harmful_safe"] == "Yes"))
df = df[~condition]

df['prefix_prompt'] = df.apply(
    lambda row: correct_prefix(row['prefix_prompt'], row['sensitive_word']),
    axis=1
)

df['suffix_prompt'] = df.apply(
    lambda row: correct_suffix(row['suffix_prompt'], row['sensitive_word']),
    axis=1
)

df.to_csv("NEW_outputs_corrected.csv", index=False)