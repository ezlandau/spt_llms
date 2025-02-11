from typing import List
import pandas as pd

from deepeval.benchmarks import HumanEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks.schema import MultipleChoiceSchema, NumberSchema
from transformers import AutoTokenizer, AutoModelForCausalLM


class CustomLM(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, *args, **kwargs) -> MultipleChoiceSchema:
        model = self.load_model()
        device = "cuda"
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)
        # Generate up to 100 tokens for code generation tasks
        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        output = self.tokenizer.batch_decode(generated_ids)[0]
        return MultipleChoiceSchema(answer=output)

    async def a_generate(self, prompt: str) -> MultipleChoiceSchema:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str], *args, **kwargs) -> List[NumberSchema]:
        model = self.load_model()
        device = "cuda"
        model_inputs = self.tokenizer(prompts, return_tensors="pt").to(device)
        model.to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return [NumberSchema(answer=self.tokenizer.decode(generated_id)) for generated_id in generated_ids]

    def get_model_name(self):
        return "CustomLM"


# Load model and tokenizer from the pretrained directory.
model = AutoModelForCausalLM.from_pretrained(
    "/storage/ukp/shared/shared_model_weights/models--google--gemma-2-9b-it"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/storage/ukp/shared/shared_model_weights/models--google--gemma-2-9b-it"
)

gemma2 = CustomLM(model=model, tokenizer=tokenizer)

# Set up and evaluate the HumanEval benchmark with 5 in-context examples.
benchmark_humaneval_gemma2 = HumanEval(n_shots=5)
results_humaneval_gemma2 = benchmark_humaneval_gemma2.evaluate(model=gemma2)

# Save the predictions and overall result to CSV files.
benchmark_humaneval_gemma2.predictions.to_csv("./humaneval-gemma2-eval.csv")
pd.DataFrame({'Metric': ["result"], 'Value': [results_humaneval_gemma2]}).to_csv(
    './humaneval-gemma2-result.csv', index=False
)
