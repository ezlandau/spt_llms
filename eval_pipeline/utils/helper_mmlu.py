import re

import pandas as pd
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.schema import MultipleChoiceSchema
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoTokenizer

from config_eval import ConvTemplates
from language_models import LLM as CustomLLM
from smoothllm import SmoothLLM, Prompt

SYSTEM_PROMPTS = {
    "gemma2-9b": (
        "Answer with a single letter: A, B, C, or D. Do not include any punctuation or extra text."
    ),
    "phi-4-14b": (
        "Answer with a single letter: A, B, C, or D. Do not include any punctuation or extra text."
    ),
    "llama31-8b": (
        "Answer with a single letter: A, B, C, or D. Do not include any punctuation or extra text."
    )
}

def create_prompt(prompt: str, model_name: str, output: str = None) -> str:
    """
    Create a formatted prompt based on the model's requirements.
    """
    system_prompt = SYSTEM_PROMPTS.get(model_name, "")
    if model_name in {"phi-4-14b", "llama31-8b"}:
        if model_name == "phi-4-14b":
            if system_prompt:
                template = (
                    f"<|system|>\n{system_prompt}<|end|>\n"
                    f"<|user|>\n{prompt}<|end|>\n"
                    f"<|assistant|> "
                )
            else:
                template = (
                    f"<|user|>\n{prompt}<|end|>\n"
                    f"<|assistant|> "
                )
        else:  # llama31-8b
            if system_prompt:
                template = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|> "
                )
            else:
                template = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|> "
                )
        return template.strip()

    # Default conversation template for other models.
    conv = ConvTemplates().get_template(model_name)
    if system_prompt:
        conv.system_message = system_prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], output or "")
    return conv.get_prompt()

class CustomLM(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    # SmoothLLM generate
    def generate(self, prompt: str, *args, **kwargs) -> MultipleChoiceSchema:

        smoothllm_instance = SmoothLLM(
            target_model=self.model,
            pert_type='RandomSwapPerturbation',
            pert_pct=10,
            num_copies=10
        )
        full_prompt = Prompt(create_prompt(prompt, "phi-4-14b"), prompt, 2)
        _, _, output = smoothllm_instance(full_prompt)
        decoded_output = output

        # Extract last two tokens
        last_token = decoded_output[-1].upper()
        prev_token = decoded_output[-2].upper() if len(decoded_output) > 1 else ""

        # Match against A, B, C, D
        choice_regex = re.compile(r"^[ABCD]$")
        if choice_regex.match(last_token):
            selected_token = last_token
        elif choice_regex.match(prev_token):
            selected_token = prev_token
        else:
            selected_token = "A"
        return MultipleChoiceSchema(answer=selected_token)

    # # Default generate
    # def generate(self, prompt: str, *args, **kwargs) -> MultipleChoiceSchema:
    #     model = self.load_model()
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    #     model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
    #     model.to(device)
    #
    #     generated_ids = model.generate(**model_inputs, max_new_tokens=2, do_sample=True)
    #     decoded_output = self.tokenizer.batch_decode(generated_ids)[0].strip().split()
    #
    #     # Extract last two tokens
    #     last_token = decoded_output[-1].upper()
    #     prev_token = decoded_output[-2].upper() if len(decoded_output) > 1 else ""
    #
    #     # Match against A, B, C, D
    #     choice_regex = re.compile(r"^[ABCD]$")
    #     if choice_regex.match(last_token):
    #         selected_token = last_token
    #     elif choice_regex.match(prev_token):
    #         selected_token = prev_token
    #     else:
    #         selected_token = "A"  # Default fallback
    #
    #     return MultipleChoiceSchema(answer=selected_token)

    async def a_generate(self, prompt: str) -> MultipleChoiceSchema:
        return self.generate(prompt)

    def get_model_name(self):
        return "CustomLM"

conv_templates = ConvTemplates()

model = CustomLLM(
            model_path="/ukp-storage-1/zadorin/spt_llms/weights/phi-4",
            tokenizer_path="/ukp-storage-1/zadorin/spt_llms/weights/phi-4",
            conv_template_name=conv_templates.get_template_name("gemma2-9b"),
            device='cuda:0'
        )
tokenizer = AutoTokenizer.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/phi-4")

gemma2 = CustomLM(model, tokenizer=tokenizer)

benchmark_mmlu_gemma2 = MMLU(
    n_shots=5,
    tasks=[MMLUTask.HIGH_SCHOOL_EUROPEAN_HISTORY, MMLUTask.HIGH_SCHOOL_US_HISTORY, MMLUTask.GLOBAL_FACTS, MMLUTask.ANATOMY,
           MMLUTask.COLLEGE_BIOLOGY, MMLUTask.COMPUTER_SECURITY]
)

results_mmlu_gemma2 = benchmark_mmlu_gemma2.evaluate(model=gemma2)
benchmark_mmlu_gemma2.predictions.to_csv("./mmlu-smoothllm-phi4-eval.csv")
pd.DataFrame({'Metric': ["result"], 'Value': [results_mmlu_gemma2]}).to_csv('./mmlu-smoothllm-phi4-result.csv', index=False)

#
# model = AutoModelForCausalLM.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/Llama-3.1-8B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/Llama-3.1-8B-Instruct")
#
# gemma2 = CustomLM(model=model, tokenizer=tokenizer)
#
# benchmark_mmlu_gemma2 = MMLU(
#      n_shots=5,
# )
#
# results_mmlu_gemma2 = benchmark_mmlu_gemma2.evaluate(model=gemma2)
# benchmark_mmlu_gemma2.predictions.to_csv("./mmlu-llama31-eval.csv")
# pd.DataFrame({'Metric': ["result"], 'Value': [results_mmlu_gemma2]}).to_csv('./mmlu-llama31-result.csv', index=False)
#
# model = AutoModelForCausalLM.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/phi-4")
# tokenizer = AutoTokenizer.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/phi-4")
#
# gemma2 = CustomLM(model=model, tokenizer=tokenizer)
#
# benchmark_mmlu_gemma2 = MMLU(
#      n_shots=5,
# )
#
# results_mmlu_gemma2 = benchmark_mmlu_gemma2.evaluate(model=gemma2)
# benchmark_mmlu_gemma2.predictions.to_csv("./mmlu-phi-4-eval.csv")
# pd.DataFrame({'Metric': ["result"], 'Value': [results_mmlu_gemma2]}).to_csv('./mmlu-phi-4-result.csv', index=False)