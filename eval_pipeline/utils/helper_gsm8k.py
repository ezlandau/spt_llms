from transformers import AutoTokenizer, AutoModelForCausalLM
import re

import pandas as pd
from deepeval.benchmarks import GSM8K
from deepeval.benchmarks.schema import NumberSchema
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM


class CustomLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, *args, **kwargs) -> NumberSchema:
        model = self.load_model()

        device = "cuda"

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=1300, temperature=0.1, do_sample=True)
        match = re.search(r'\*\*Answer\*\*: \n\nLet\'s think step-by-step[\s\S]*?(\d+)\.*\s*(?=\n\n|\*\*Problem\*\*|<\|endoftext\|>)',
                              self.tokenizer.batch_decode(generated_ids)[0], re.DOTALL)
        return NumberSchema(answer=match.group(1))

    async def a_generate(self, prompt: str) -> NumberSchema:
        return self.generate(prompt)

    def get_model_name(self):
        return "CustomLM"

model = AutoModelForCausalLM.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/phi-4")
tokenizer = AutoTokenizer.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/phi-4")

phi4 = CustomLM(model=model, tokenizer=tokenizer)

benchmark_gsm8k = GSM8K(
    n_problems=1319,
    n_shots=8,
    enable_cot=True
)

results_gsm8k = benchmark_gsm8k.evaluate(model=phi4)
benchmark_gsm8k.predictions.to_csv("./gsm8k-phi4-eval.csv")
pd.DataFrame({'Metric': ["result"], 'Value': [results_gsm8k]}).to_csv('./gsm8k-phi4-result.csv', index=False)
