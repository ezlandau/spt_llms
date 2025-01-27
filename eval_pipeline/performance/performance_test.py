from typing import List
from vllm import LLM, SamplingParams
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU
from deepeval.benchmarks import GSM8K
import re
import pandas as pd


class Gemma29B(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def _format_prompt(self, prompt: str) -> str:
        """Formats any incoming prompt into LLaMA 3's chat template."""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Answer the following multiple-choice question. Provide only the letter of the correct answer.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{prompt}
Answer: <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    def generate(self, prompt: str) -> str:
        formatted_prompt = self._format_prompt(prompt)
        print("Formatted prompt:", formatted_prompt)  # Debug
        model = self.load_model()

        sampling_params = SamplingParams(
            max_tokens=20,  # Increased to ensure answer is included
            temperature=0,
            top_p=1.0,
            top_k=1,
            stop=["<|eot_id|>"]  # Stop at LLaMA's end-of-turn token
        )

        outputs = model.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        # Extract first occurrence of A-D (case-insensitive)
        match = re.search(r'\b([A-Da-d])\b', generated_text, re.IGNORECASE)
        print("Raw output:", generated_text)  # Debug
        return match.group(1).upper() if match else generated_text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        formatted_prompts = [self._format_prompt(p) for p in prompts]
        model = self.load_model()
        sampling_params = SamplingParams(
            max_tokens=20,
            temperature=0,
            top_p=1.0,
            top_k=1,
            stop=["<|eot_id|>"]
        )
        outputs = model.generate(formatted_prompts, sampling_params)
        return [self._extract_answer(output.outputs[0].text) for output in outputs]

    def _extract_answer(self, text: str) -> str:
        text = text.strip()
        match = re.search(r'\b([A-Da-d])\b', text, re.IGNORECASE)
        return match.group(1).upper() if match else text

    def get_model_name(self):
        return "Gemma 2-9B"


# Initialize model with correct prompt formatting
def format_mmlu_prompt(question: str, options: List[str]) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Answer the following multiple-choice question. Provide only the letter of the correct answer.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{question}
Options:
{options}
Answer: <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# llama31_8b = Llama318B(model=LLM(model="/storage/ukp/shared/shared_model_weights/models--llama-3.1/hf/8B-Instruct"))
gemma2_9b = Gemma29B(model=LLM(model="/storage/ukp/shared/shared_model_weights/models--google--gemma-2-9b-it"))
# phi_4 = Llama318B(model=LLM(model="/ukp-storage-1/zadorin/spt_llms/weights/phi-4"))

# Test with formatted prompt
# formatted_prompt = format_mmlu_prompt(
#     "What is the capital of France?",
#     "A. London\nB. Paris\nC. Berlin\nD. Madrid"
# )
# print(llama31_8b.generate(formatted_prompt))  # Should output "B"
#
# benchmark_mmlu1 = MMLU(n_shots=5)
#
# results_llama = benchmark_mmlu1.evaluate(model=llama31_8b)
# benchmark_mmlu1.predictions.to_csv("./predictions_mmlu_llama.csv")
# pd.DataFrame({'Metric': ["result"], 'Value': [results_llama]}).to_csv('./results_mmlu_llama.csv', index=False)

benchmark_mmlu2 = MMLU(n_shots=5)

results_gemma = benchmark_mmlu2.evaluate(model=gemma2_9b)
benchmark_mmlu2.predictions.to_csv("./predictions_mmlu_gemma.csv")
pd.DataFrame({'Metric': ["result"], 'Value': [results_gemma]}).to_csv('./results_mmlu_gemma.csv', index=False)

# benchmark_mmlu3 = MMLU(n_shots=5)
#
# results_phi = benchmark_mmlu3.evaluate(model=phi_4)
# benchmark_mmlu3.predictions.to_csv("./predictions_mmlu_phi.csv")
# pd.DataFrame({'Metric': ["result"], 'Value': [results_phi]}).to_csv('./results_mmlu_phi.csv', index=False)


benchmark_gsm8k = GSM8K(
    n_problems=1319,
    n_shots=5,
    enable_cot=True
)

results_gemma = benchmark_gsm8k.evaluate(model=gemma2_9b)
benchmark_gsm8k.predictions.to_csv("./predictions_gsm8k_gemma.csv")
pd.DataFrame({'Metric': ["result"], 'Value': [results_gemma]}).to_csv('./results_gsm8k_gemma.csv', index=False)

# Overall MMLU Accuracy: 0.5561885771257655 for Llama 31-8b
# Overall GSM8K Accuracy: 0.123 for Llama 31-8b
