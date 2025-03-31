import re
import pandas as pd


def parse_record(record):
    """
    Parse a single record (block of text separated by dashes) and return a list of rows.
    Each row is a dictionary with keys:
    sensitive_word, benign_prompt, harmful_prompt, suffix_prompt, prefix_prompt.

    If a record has multiple sets (e.g. eight prompts), then split into multiple rows.
    """
    rows = []
    # Get the sensitive word line
    sw_match = re.search(r'^Sensitive Word:\s*(.+)$', record, flags=re.MULTILINE)
    if not sw_match:
        return rows
    # The sensitive word line may contain multiple words separated by commas.
    # We keep the whole string here (later, if needed, you could split it).
    sensitive_word_full = sw_match.group(1).strip()
    # Remove extra quotes if present
    sensitive_word_full = sensitive_word_full.strip('"').strip("'")

    # Find all prompt lines that follow the pattern: "number. Label: prompt text"
    # This regex captures the order number, label, and prompt text.
    prompt_pattern = re.compile(r'^\d+\.\s*(.+?):\s*(.+)$', re.MULTILINE)
    prompts = prompt_pattern.findall(record)

    # If the number of prompts is not a multiple of 4, try to use only complete groups.
    n = len(prompts)
    if n < 4:
        return rows
    groups = n // 4  # number of complete groups
    for i in range(groups):
        group = prompts[i * 4:(i + 1) * 4]
        # Expecting the order to be:
        # 1. Benign Logical Prompt, 2. Really Harmful Prompt,
        # 3. Benign Random Prompt with Suffix, 4. Benign Random Prompt with Prefix
        benign_label, benign_prompt = group[0]
        harmful_label, harmful_prompt = group[1]
        suffix_label, suffix_prompt = group[2]
        prefix_label, prefix_prompt = group[3]

        row = {
            'sensitive_word': sensitive_word_full,
            'benign_prompt': benign_prompt.strip(),
            'harmful_prompt': harmful_prompt.strip(),
            'suffix_prompt': suffix_prompt.strip(),
            'prefix_prompt': prefix_prompt.strip()
        }
        rows.append(row)
    return rows


def analyze_prompts(row):
    """
    Perform a basic analysis:
      - For the prefix prompt: check if it starts with the sensitive word (ignoring case and punctuation).
      - For the suffix prompt: check if it ends with the sensitive word.
      - For harmful/benign prompts, check for indicative keywords.
    Returns a dictionary with analysis comments.
    """
    analysis = {}
    sw = row['sensitive_word'].lower()

    # Clean a string: remove quotes and extra punctuation from beginning/end.
    def clean(s):
        return s.strip(' "\'.,').lower()

    prefix_clean = clean(row['prefix_prompt'])
    suffix_clean = clean(row['suffix_prompt'])

    analysis['prefix_correct'] = prefix_clean.startswith(sw)
    analysis['suffix_correct'] = suffix_clean.endswith(sw)

    # Check harmful prompt for keywords (this is a simple heuristic)
    harmful_keywords = ['kill', 'harm', 'illegal', 'incite', 'execute', 'steal', 'attack', 'exploit', 'break in']
    if any(kw in row['harmful_prompt'].lower() for kw in harmful_keywords):
        analysis['harmful_prompt_flagged'] = True
    else:
        analysis['harmful_prompt_flagged'] = False

    # Check benign prompt – we expect a more neutral, non-inciting tone.
    benign_negative = ['kill', 'harm', 'incite', 'illegal', 'attack', 'execute']
    if any(kw in row['benign_prompt'].lower() for kw in benign_negative):
        analysis['benign_prompt_flagged'] = "Potential issue: contains harmful terms."
    else:
        analysis['benign_prompt_flagged'] = "Seems neutral."

    return analysis


def main():
    # Read the outputs.txt file
    with open('outputs.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Split records by a line of dashes (at least 10 dashes)
    records = re.split(r'-{10,}', content)

    all_rows = []
    analysis_results = []

    for record in records:
        record = record.strip()
        if not record:
            continue
        rows = parse_record(record)
        for row in rows:
            all_rows.append(row)
            analysis_results.append(analyze_prompts(row))

    # Create a DataFrame from the parsed rows
    df = pd.DataFrame(all_rows,
                      columns=['sensitive_word', 'benign_prompt', 'harmful_prompt', 'prefix_prompt', 'suffix_prompt'])

    # Optionally, create a DataFrame for analysis results (merged with the original data)
    analysis_df = pd.DataFrame(analysis_results)
    df_analysis = pd.concat([df, analysis_df], axis=1)

    # Save the CSV
    df_analysis.to_csv('parsed_outputs.csv', index=False)
    print("CSV file 'parsed_outputs.csv' has been created with the following columns:")
    print(df_analysis.columns.tolist())
    print("\nAnalysis summary:")
    print(df_analysis[['sensitive_word', 'prefix_correct', 'suffix_correct', 'harmful_prompt_flagged',
                       'benign_prompt_flagged']])


if __name__ == '__main__':
    main()

