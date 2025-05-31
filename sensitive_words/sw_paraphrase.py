import os
import time
import logging
import random

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import pandas as pd
from collections import Counter
import en_core_web_sm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, SequenceBiasLogitsProcessor

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
logger.info(f"Set random seed to {SEED} for reproducibility")

MODEL_NAME = "/ukp-storage-1/zadorin/spt_llms/sensitive_words/models--mistralai--Mixtral-8x7B-Instruct-v0.1"

def get_max_memory():
    if not torch.cuda.is_available():
        return None
    total_gpus = torch.cuda.device_count()
    per_gpu = int(79)
    return {i: f"{per_gpu}GiB" for i in range(total_gpus)}

logger.info(f"Loading model and tokenizer from {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    max_memory=get_max_memory(),
)

EXCLUDE_THRESHOLD = 90
FORBIDDEN_STARTS = set()
START_COUNTS     = Counter()

QUESTION_WORDS = ["How", "What", "Which", "Write", "Explain", "Why", "List",
                  "Describe", "Outline", "Tell", "Show", "Find", "Who", "Analyze",
                  "Identify", "Explore", "Whom", "Can"]


def is_valid_prompt(text: str, nlp) -> bool:
    doc = nlp(text)
    if not doc:
        return False
    has_verb = doc[0].pos_ == "VERB"
    ends_q  = text.rstrip().endswith("?")
    return has_verb or ends_q


def call_model(prompt: str, forced_word: str, max_new_tokens: int = 64) -> list[str]:
    """
    Generates several candidates, biasing the exact `forced_word` token sequence
    to appear via SequenceBiasLogitsProcessor.
    """
    logger.info("Calling model for word '%s'...", forced_word)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    word_ids = tuple(tokenizer.encode(forced_word, add_special_tokens=False))
    bias_value = 10.0
    sequence_bias = {word_ids: bias_value}

    logits_processor = LogitsProcessorList([
        SequenceBiasLogitsProcessor(sequence_bias=sequence_bias)
    ])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        num_return_sequences=3,
        logits_processor=logits_processor,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    candidates = []
    for seq in outputs:
        gen = seq[inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        text = text.split("\n")[0].strip(' "')
        candidates.append(text)
    return candidates


def paraphrase(orig: str, word: str, nlp, max_tries: int = 10) -> str:
    qws = QUESTION_WORDS
    instr = (
        "You’re a rewrite engine. Produce exactly one paraphrase that:\n"
        f"– begins with one of: {', '.join(qws)}\n"
        f"– preserves the exact sensitive word: “{word}”\n"
        "– rewrites everything else in different words and structure\n"
        "– remains a valid harmful prompt with no extra commentary"
    )
    prompt = f"{instr}\nSensitive word: \"{word}\"\nHarmful prompt: \"{orig}\"\nParaphrase:"

    for attempt in range(1, max_tries + 1):
        candidates = call_model(prompt, forced_word=word)
        for txt in candidates:
            if word not in txt:
                continue
            if is_valid_prompt(txt, nlp):
                first = txt.split()[0]
                START_COUNTS[first] += 1
                if START_COUNTS[first] > EXCLUDE_THRESHOLD:
                    FORBIDDEN_STARTS.add(first)
                print(f"{word!r} -> {txt!r}")
                return txt
    return orig


def main():
    try:
        nlp = en_core_web_sm.load()
    except OSError:
        logger.warning("Could not load en_core_web_sm; falling back to blank English model")
        import spacy
        nlp = spacy.blank("en")

    df = pd.read_csv("sensitive_words.csv")
    subset = df.loc[1020:, ["sensitive_word", "harmful_prompt"]]

    for idx, row in subset.iterrows():
        orig = row["harmful_prompt"]
        paraphrased = paraphrase(orig, row["sensitive_word"], nlp)
        df.at[idx, "harmful_prompt"] = paraphrased
        time.sleep(1)

    df.to_csv("sensitive_words.tmp.csv", index=False)

if __name__ == "__main__":
    main()
