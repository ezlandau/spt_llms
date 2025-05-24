import re
from collections import Counter

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval.benchmarks import GSM8K
from deepeval.benchmarks.schema import NumberSchema
from deepeval.models.base_model import DeepEvalBaseLLM

from transformers import StoppingCriteria, StoppingCriteriaList

class StopAfterAnswer(StoppingCriteria):
    def __init__(self, tokenizer, stop_seq="Answer:"):
        self.tokenizer = tokenizer
        self.stop_ids = tokenizer.encode(stop_seq, add_special_tokens=False)
    def __call__(self, input_ids, scores, **kwargs):
        # Check if last tokens match "Answer:"
        if input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids:
            return True
        return False

SYSTEM_PROMPTS = {
    "llama31-8b": (
        "Given the following 8 math problems as examples, solve problem 9 by thinking step by step, "
        "showing all your work.  After your reasoning, prefix your final answer with “Answer:” and stop.  "
    )
}

class CustomLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model_id: str,
        tokenizer_id: str,
        model_name: str,
        device: str = "cuda:0",
    ):
        self.device = device
        self.model_name = model_name

        tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False)
        if isinstance(tok, bool):
            tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)

        if getattr(tok, "pad_token", None) is None:
            tok.add_special_tokens({"pad_token": tok.eos_token})
        if hasattr(tok, "padding_side"):
            tok.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
        )
        model.resize_token_embeddings(len(tok))

        self.tokenizer = tok
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str, *_, **__) -> NumberSchema:
        lm = self.load_model()
        dev = next(lm.parameters()).device

        sys_p = SYSTEM_PROMPTS.get(self.model_name, "")
        full_prompt = f"{sys_p}\n\n{prompt}\nAfter your reasoning, prefix your final answer with 'Answer:' and stop generation. Let's think step by step."

        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True).to(dev)

        # 1) sample 10 full chains
        outputs = lm.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            bos_token_id=128000,
            eos_token_id=[128001, 128009],
            top_p=0.9,
        )

        flag = True
        answers = []
        input_length = inputs["input_ids"].shape[-1]
        for seq in outputs:
            gen_tokens = seq[input_length:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            print("GENERATED ONLY:", text, "\n\n")
            # 2) regex-extract the first number after "Answer:"
            pattern = re.compile(
                r"answer\b.*?(-?\d+(?:\.\d+)?)",
                re.IGNORECASE | re.DOTALL
            )
            m = pattern.search(text)
            if m:
                val_str = m.group(1)
                print("EXTRACTED:", val_str, "\n\n")
                parsed = int(val_str) if val_str.isdigit() else int(float(val_str))
                answers.append(parsed)
            else:
                answers.append(0)

        # 4) vote for the most common
        most_common, _ = Counter(answers).most_common(1)[0]

        # 5) wrap and return
        return NumberSchema(answer=int(most_common))

    async def a_generate(self, prompt: str) -> NumberSchema:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


if __name__ == "__main__":
    llama31 = CustomLM(
        model_id="ez-landau/SFT-SW-Llama-3.1-8B-Instruct-NOTEST_LORA_SW",
        tokenizer_id="ez-landau/SFT-SW-Llama-3.1-8B-Instruct-NOTEST_LORA_SW",
        model_name="llama31-8b",
        device="cuda:0",
    )

    benchmark = GSM8K(
        n_problems=1000,
        n_shots=8,
        enable_cot=True,
    )
    result = benchmark.evaluate(model=llama31)

    benchmark.predictions.to_csv("./gsm8k-llama31-8b-lora-sft-eval.csv", index=False)
    pd.DataFrame({"Metric": ["GSM8K"], "Value": [result]}).to_csv(
        "../perfevals/gsm8k-llama31-8b-lora-sft-result.csv", index=False
    )
