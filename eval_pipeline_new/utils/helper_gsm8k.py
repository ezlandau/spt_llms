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

        generated_ids = model.generate(**model_inputs, max_new_tokens=1024, temperature=0.1, do_sample=True)
        match = re.search(r'\*\*Answer\*\*: \n\nLet\'s think step-by-step.*?(\d+)\.*\s*(?=\n\n|\*\*Problem\*\*)',
                              self.tokenizer.batch_decode(generated_ids)[0], re.DOTALL)
        return NumberSchema(answer=match.group(1))

    async def a_generate(self, prompt: str) -> NumberSchema:
        return self.generate(prompt)

    def get_model_name(self):
        return "CustomLM"

model = AutoModelForCausalLM.from_pretrained("/storage/ukp/shared/shared_model_weights/models--google--gemma-2-9b-it")
tokenizer = AutoTokenizer.from_pretrained("/storage/ukp/shared/shared_model_weights/models--google--gemma-2-9b-it")

llama3 = CustomLM(model=model, tokenizer=tokenizer)

benchmark_gsm8k_gemma2 = GSM8K(
    n_problems=1319,
    n_shots=8,
    enable_cot=True
)

results_gsm8k_gemma2 = benchmark_gsm8k_gemma2.evaluate(model=llama3)
benchmark_gsm8k_gemma2.predictions.to_csv("./gsm8k-gemma2-eval.csv")
pd.DataFrame({'Metric': ["result"], 'Value': [results_gsm8k_gemma2]}).to_csv('./gsm8k-gemma2-result.csv', index=False)
