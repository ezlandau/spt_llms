import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from tqdm import tqdm
from util.gsm8k_helper import nshot_chats, extract_ans_from_response
from util.json_io import load_jsonlines
import transformers
import torch

model_path = "/storage/ukp/shared/shared_model_weights/models--llama-3/8B-Instruct"

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
)

def get_response(chats):
    gen_text = generator(chats)[0]
    return gen_text['generated_text'][-1]['content']

train_data = load_jsonlines('data/gsm8k/train.jsonl')
test_data = load_jsonlines('data/gsm8k/test.jsonl')

N_SHOT = 8
messages = nshot_chats(nshot_data=train_data, n=N_SHOT, question=test_data[0]['question'])

response = get_response(messages)
print(response)

pred_ans = extract_ans_from_response(response)
print("FINAL ANSWER:\n", pred_ans)

ground_truth = test_data[0]['answer']

true_ans = extract_ans_from_response(ground_truth)
print("GROUND TRUTH\n", true_ans)

log_file_path = './log/errors.txt'
with open(log_file_path, 'w') as log_file:
    log_file.write('')

total = correct = 0
for qna in tqdm(test_data):

    messages = nshot_chats(nshot_data=train_data, n=N_SHOT, question=qna['question'])
    response = get_response(messages)

    pred_ans = extract_ans_from_response(response)
    true_ans = extract_ans_from_response(qna['answer'])

    total += 1
    if pred_ans != true_ans:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{messages}\n\n")
            log_file.write(f"Response: {response}\n\n")
            log_file.write(f"Ground Truth: {qna['answer']}\n\n")
            log_file.write(f"Current Accuracy: {correct / total:.3f}\n\n")
            log_file.write('\n\n')
    else:
        correct += 1

print(f"Total Accuracy: {correct / total:.3f}")
# from typing import List
# from vllm import LLM, SamplingParams
# from deepeval.models.base_model import DeepEvalBaseLLM
# from deepeval.benchmarks.tasks import MMLUTask
# from deepeval.benchmarks import MMLU
# from deepeval.benchmarks import GSM8K
# from deepeval.benchmarks import HumanEval
# import re
# import pandas as pd
# import fastchat
#
# class Phi4(DeepEvalBaseLLM):
#     def __init__(self, model):
#         self.model = model
#         self.benchmark = None
#
#     def load_model(self):
#         return self.model
#
#     def _format_prompt(self, prompt: str) -> str:
#         """Formats any incoming prompt into chat template."""
#         templates = {
#             "MMLU": lambda p: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     Answer the following multiple-choice question. Provide only the letter of the correct answer.<|eot_id|>
#     <|start_header_id|>user<|end_header_id|>
#     {p}
#     Answer: <|eot_id|>
#     <|start_header_id|>assistant<|end_header_id|>
#     """,
#             "GSM8K": lambda p: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     Solve the following math problem step by step. Provide detailed reasoning and include the final answer at the end.<|eot_id|>
#     <|start_header_id|>user<|end_header_id|>
#     {p}
#     The answer is: <|eot_id|>
#     <|start_header_id|>assistant<|end_header_id|>
#     """,
#           "HumanEval": lambda p: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     Complete the following Python function based on the provided docstring and test cases. Provide only the function implementation.<|eot_id|>
#     <|start_header_id|>user<|end_header_id|>
#     {p}
#     Solution: <|eot_id|>
#     <|start_header_id|>assistant<|end_header_id|>
#     """
#         }
#         # Get the corresponding lambda, defaulting to an empty string if benchmark not found
#         formatted_prompt = templates.get(self.benchmark, lambda p: "")(prompt)
#         return formatted_prompt
#
#     def generate(self, prompt: str) -> str:
#         formatted_prompt = prompt #        self._format_prompt(prompt)
#         # print("Formatted prompt:", formatted_prompt)  # Debug
#         model = self.load_model()
#
#         if self.benchmark == "MMLU":
#             sampling_params = SamplingParams(
#                 max_tokens=20,
#                 temperature=0,
#                 top_p=1.0,
#                 top_k=1,
#                 stop=["<|eot_id|>"]
#             )
#         # https://github.com/meta-llama/llama/issues/325
#         elif self.benchmark == "GSM8K":
#             print("Processing GSM8K")
#             sampling_params = SamplingParams(
#                 max_tokens=512,
#                 temperature=0.8,
#                 top_p=0,
#                 top_k=1
#             )
#
#         outputs = model.generate([formatted_prompt], sampling_params)
#         generated_text = outputs[0].outputs[0].text.strip()
        print("Raw output:\n", outputs)  # Debug

        if self.benchmark == "MMLU":
            match = re.search(r'\b([A-D])\b', generated_text)
            return match.group(1).upper() if match else generated_text
        if self.benchmark == "GSM8K":
            match = re.search(r'The answer is (-?[0-9.,]*[0-9]+)', generated_text)
            return match.group(1) if match else generated_text
#
#     async def a_generate(self, prompt: str) -> str:
#         return self.generate(prompt)
#
#     def batch_generate(self, prompts: List[str]) -> List[str]:
#         formatted_prompts = [self._format_prompt_mmlu(p) for p in prompts]
#         model = self.load_model()
#         sampling_params = SamplingParams(
#             max_tokens=20,
#             temperature=0,
#             top_p=1.0,
#             top_k=1,
#             stop=["<|eot_id|>"]
#         )
#         outputs = model.generate(formatted_prompts, sampling_params)
#         return [self._extract_answer(output.outputs[0].text) for output in outputs]
#
#     def _extract_answer(self, text: str) -> str:
#         text = text.strip()
#         match = re.search(r'\b([A-Da-d])\b', text, re.IGNORECASE)
#         return match.group(1).upper() if match else text
#
#     def get_model_name(self):
#         return "Phi-4 14B"
#
# phi4 = Phi4(model=LLM(model="/ukp-storage-1/zadorin/spt_llms/weights/phi-4"))
#
# # phi4.benchmark = "MMLU"
# #
# # # Test with formatted prompt
# # test_prompt_mmlu = """As of 2016, about what percentage of adults aged 18 years or older were overweight?
# # A) 10%
# # B) 20%
# # C) 40%
# # D) 80%"""
# #
# # # Test MMLU generation
# # print(phi4.generate(test_prompt_mmlu))  # Should output "C"
#
# # # MMLU Generation on Global Facts
# # benchmark_mmlu_phi4 = MMLU(
# #     n_shots=5,
# #     tasks=[MMLUTask.GLOBAL_FACTS]
# # )
# #
# # phi4.benchmark = "MMLU"
# # results_mmlu_phi4 = benchmark_mmlu_phi4.evaluate(model=phi4)
# # benchmark_mmlu_phi4.predictions.to_csv("./predictions_mmlu_phi4.csv")
# # pd.DataFrame({'Metric': ["result"], 'Value': [results_mmlu_phi4]}).to_csv('./results_mmlu_phi4.csv', index=False)
#
# phi4.benchmark = "GSM8K"
# #
# # test_prompt_gsm8k = """Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins
# # for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
# # How much in dollars does she make every day at the farmers' market?"""
# # # Test GSM8K generation
# # print(phi4.generate(test_prompt_gsm8k)) # Should output "18"
#
# benchmark_gsm8k_phi4 = GSM8K(
#     n_problems=1,
#     n_shots=5,
#     enable_cot=True
# )
#
# results_gsm8k_phi4 = benchmark_gsm8k_phi4.evaluate(model=phi4)
# benchmark_gsm8k_phi4.predictions.to_csv("./predictions_gsm8k_phi4.csv")
# pd.DataFrame({'Metric': ["result"], 'Value': [results_gsm8k_phi4]}).to_csv('./results_gsm8k_phi4.csv', index=False)
#
#
# # results_llama = benchmark_gsm8k.evaluate(model=llama31_8b)
# # benchmark_gsm8k.predictions.to_csv("./predictions_gsm8k_lllama.csv")
# # pd.DataFrame({'Metric': ["result"], 'Value': [results_llama]}).to_csv('./results_gsm8k_llama.csv', index=False)
#
# # gemma2_9b = Llama318B(model=LLM(model="/storage/ukp/shared/shared_model_weights/models--google--gemma-2-9b-it"))
# # phi_4 = Llama318B(model=LLM(model="/ukp-storage-1/zadorin/spt_llms/weights/phi-4"))
# # benchmark_mmlu1 = MMLU(n_shots=5)
# #
# # results_llama = benchmark_mmlu1.evaluate(model=llama31_8b)
#
#
# # benchmark_mmlu2 = MMLU(n_shots=5)
# #
# # results_gemma = benchmark_mmlu2.evaluate(model=gemma2_9b)
# # benchmark_mmlu2.predictions.to_csv("./predictions_mmlu_gemma.csv")
# # pd.DataFrame({'Metric': ["result"], 'Value': [results_gemma]}).to_csv('./results_mmlu_gemma.csv', index=False)
# #
# # benchmark_mmlu3 = MMLU(n_shots=5)
# #
# # results_phi = benchmark_mmlu3.evaluate(model=phi_4)
# # benchmark_mmlu3.predictions.to_csv("./predictions_mmlu_phi.csv")
# # pd.DataFrame({'Metric': ["result"], 'Value': [results_phi]}).to_csv('./results_mmlu_phi.csv', index=False)
#
# # Overall MMLU Accuracy: 0.5561885771257655 for Llama 31-8b
# # Overall GSM8K Accuracy: 0.123 for Llama 31-8b
