from typing import List

import pandas as pd
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.benchmarks.schema import MultipleChoiceSchema, NumberSchema
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

    def generate(self, prompt: str, *args, **kwargs) -> MultipleChoiceSchema:
        model = self.load_model()

        device = "cuda"

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=2, do_sample=True)
        return MultipleChoiceSchema(answer=tokenizer.batch_decode(generated_ids)[0][-1].capitalize())

    async def a_generate(self, prompt: str) -> MultipleChoiceSchema:
        return self.generate(prompt)

    # This is optional.
    def batch_generate(self, prompts: List[str], *args, **kwargs) -> List[NumberSchema]:
        model = self.load_model()
        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer(prompts, return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=1, do_sample=True)
        return [NumberSchema(answer=self.tokenizer.decode(generated_id)[-1]) for generated_id in generated_ids]

    def get_model_name(self):
        return "CustomLM"

model = AutoModelForCausalLM.from_pretrained("/storage/ukp/shared/shared_model_weights/models--google--gemma-2-9b-it")
tokenizer = AutoTokenizer.from_pretrained("/storage/ukp/shared/shared_model_weights/models--google--gemma-2-9b-it")

gemma2 = CustomLM(model=model, tokenizer=tokenizer)

benchmark_mmlu_gemma2 = MMLU(
     n_shots=5,
     tasks=[MMLUTask.GLOBAL_FACTS]
)

results_mmlu_gemma2 = benchmark_mmlu_gemma2.evaluate(model=gemma2)
benchmark_mmlu_gemma2.predictions.to_csv("./mmlu-gemma2-eval.csv")
pd.DataFrame({'Metric': ["result"], 'Value': [results_mmlu_gemma2]}).to_csv('./mmlu-gemma2-result.csv', index=False)
