import requests
import pandas as pd


def evaluate_safety(prompt, server_url="http://localhost:8000/v1/chat/completions"):
    chat = [
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "meta-llama/Llama-Guard-3-8B",
        "messages": chat
    }
    response = requests.post(server_url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    label = data["choices"][0]["message"]["content"].replace('\n', ' ').strip().split()[0]
    return "Yes" if label.lower() == "safe" else "No"


def safety_eval(csv_path):
    df = pd.read_csv(csv_path)

    df["benign_prompt"] = df["benign_prompt"].fillna("")

    df["is_benign_safe"] = df["benign_prompt"].apply(evaluate_safety)
    df["is_harmful_safe"] = df["harmful_prompt"].apply(evaluate_safety)

    df.to_csv("sensitive_words.csv", index=False)


safety_eval("sensitive_words.csv")
