import os
import logging
import pandas as pd
import spacy
import string

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CSV_PATH = "sensitive_words.csv"
PROMPT_COLUMNS = ["harmful_prompt", "benign_prompt", "prefix_prompt", "suffix_prompt"]

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy en_core_web_sm model")
except OSError:
    logger.warning("Could not load en_core_web_sm; falling back to blank English model")
    import spacy
    nlp = spacy.blank("en")
    logger.info("Loaded blank spaCy English model")

def is_valid_structure(text: str) -> bool:
    """
    Checks if the text either begins with a verb or ends with a question mark.
    """
    doc = nlp(text.strip())
    starts_with_verb = bool(doc) and doc[0].pos_ == "VERB"
    ends_with_q      = text.rstrip().endswith("?")
    return starts_with_verb or ends_with_q

def verify_prompts(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        sw = str(row["sensitive_word"]).strip()
        for col in PROMPT_COLUMNS:
            text = str(row.get(col, "")).strip()
            tokens = text.split()
            prefix = tokens[0] if tokens else ""
            raw_suffix = tokens[-1] if tokens else ""
            suffix = raw_suffix.rstrip(string.punctuation)

            if not (prefix == sw or suffix == sw):
                records.append({
                    "index": idx,
                    "column": col,
                    "sensitive_word": sw,
                    "prompt": text,
                    "error": f"SW not at prefix (‘{prefix}’) or suffix (‘{suffix}’)"
                })
                continue

            if not is_valid_structure(text):
                records.append({
                    "index": idx,
                    "column": col,
                    "sensitive_word": sw,
                    "prompt": text,
                    "error": "Does not start with VERB or end with '?'"
                })

    return pd.DataFrame(records)

def main():
    if not os.path.exists(CSV_PATH):
        logger.error(f"CSV file not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded {len(df)} rows from {CSV_PATH}")

    invalid = verify_prompts(df)
    if invalid.empty:
        logger.info("All prompts in all columns passed verification ✔️")
    else:
        logger.warning(f"Found {len(invalid)} invalid prompt entries:")
        print(invalid.to_string(index=False))
        invalid.to_csv("invalid_prompts.csv", index=False)
        logger.info("Wrote invalid entries to invalid_prompts.csv")

if __name__ == "__main__":
    main()
