import os
import pandas as pd
import time
from openai import OpenAI
from tqdm import tqdm

def generate_sft(input_path: str,
                 output_path: str,
                 api_key: str,
                 openai_model: str,
                 test_size: int = None) -> None:
    """
    Reads a CSV with a 'prompt' column, queries the OpenAI model, and writes out a new CSV
    with an added 'response' column.
    """
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(input_path)
    if test_size is not None:
        df = df.head(test_size)

    responses = []
    for prompt in tqdm(df['prompt'], desc=f"Generating responses from {os.path.basename(input_path)}"):
        try:
            completion = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            text = completion.choices[0].message.content.strip()
        except Exception as e:
            text = f"<ERROR: {e}>"
        responses.append(text)
        time.sleep(0.5)

    df['response'] = responses
    df.to_csv(output_path, index=False)
    print(f"Saved responses to {output_path}")


def generate_sft_standalone(dataset: str,
                             test_size: int = None,
                             api_key: str = "",
                             openai_model: str = "gpt-4o") -> None:
    """
    Wrapper to process one dataset file named data/{dataset}.csv
    and write to sft/{dataset}_sft.csv. Allows an optional test_size.
    """
    input_path = f"../data/{dataset}.csv"
    output_dir = "sft"
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base}_sft.csv")
    generate_sft(input_path, output_path, api_key, openai_model, test_size)


if __name__ == "__main__":
    for ds in ["sw_benign", "sw_harmful", "sw_prefix", "sw_suffix"]:
        generate_sft_standalone(ds)
    print("All datasets processed.")
