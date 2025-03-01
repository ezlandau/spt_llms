from transformers import AutoTokenizer, AutoModelForCausalLM
import re

import pandas as pd
from deepeval.benchmarks import GSM8K
from deepeval.benchmarks.schema import NumberSchema
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from config_eval import ConvTemplates
from language_models import LLM as CustomLLM
from smoothllm import SmoothLLM, Prompt

SYSTEM_PROMPTS = {
    "gemma2-9b": (
        "Given the following 8 math problems as examples, solve problem 9 step by step, following the same reasoning and answer structure as in the examples. Provide only the final numerical answer for problem 9 after **Answer**:, formatted exactly like in the examples."
    ),
    "phi-4-14b": (
        "Given the following 8 math problems as examples, solve problem 9 step by step, following the same reasoning and answer structure as in the examples. Provide only the final numerical answer for problem 9 after **Answer**:, formatted exactly like in the examples."
    ),
    "llama31-8b": (
        "Given the following 8 math problems as examples, solve problem 9 step by step, following the same reasoning and answer structure as in the examples. Provide only the final numerical answer for problem 9 after **Answer**:, formatted exactly like in the examples."
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

        smoothllm_instance = SmoothLLM(
            target_model=self.model,
            pert_type='RandomSwapPerturbation',
            pert_pct=10,
            num_copies=10
        )
        full_prompt = Prompt(create_prompt(prompt, "phi-4-14b"), prompt, 1300)
        # print("Full prompt:\n", full_prompt.full_prompt)
        _, _, output = smoothllm_instance(full_prompt)
        # print("Raw output:\n", output)
        match = re.search(r'Answer[\s\S]*?(\d+)', output, re.DOTALL)
        if match:
            return NumberSchema(answer=int(match.group(1)))
        else:
            return NumberSchema(answer=0)

        # match = re.search(r'\*\*Answer\*\*: \n\nLet\'s think step-by-step[\s\S]*?(\d+)\.*\s*(?=\n\n|\*\*Problem\*\*|<im_end>|<\|endoftext\|>)',
        #                       output, re.DOTALL)

    #
    # def generate(self, prompt: str, *args, **kwargs) -> NumberSchema:
    #     model = self.load_model()
    #
    #     device = "cuda"
    #
    #     model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
    #     model.to(device)
    #
    #     generated_ids = model.generate(**model_inputs, max_new_tokens=1300, temperature=0.1, do_sample=True)
    #     match = re.search(r'\*\*Answer\*\*: \n\nLet\'s think step-by-step[\s\S]*?(\d+)\.*\s*(?=\n\n|\*\*Problem\*\*|<\|endoftext\|>)',
    #                           self.tokenizer.batch_decode(generated_ids)[0], re.DOTALL)
    #     if match:
    #         return NumberSchema(answer=int(match.group(1)))  # Convert to integer
    #     else:
    #         return NumberSchema(answer=0)  # More explicit fallback

    async def a_generate(self, prompt: str) -> NumberSchema:
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

phi4 = CustomLM(model=model, tokenizer=tokenizer)

benchmark_gsm8k = GSM8K(
    n_problems=500,
    n_shots=8,
    enable_cot=True
)

results_gsm8k = benchmark_gsm8k.evaluate(model=phi4)
benchmark_gsm8k.predictions.to_csv("./gsm8k-smoothllm-phi4-eval.csv")
pd.DataFrame({'Metric': ["result"], 'Value': [results_gsm8k]}).to_csv('./gsm8k-smoothllm-phi4-result.csv', index=False)
#
# model = AutoModelForCausalLM.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/Llama-3.1-8B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/Llama-3.1-8B-Instruct")
#
# phi4 = CustomLM(model=model, tokenizer=tokenizer)
#
# benchmark_gsm8k = GSM8K(
#     n_problems=1319,
#     n_shots=8,
#     enable_cot=True
# )
#
# results_gsm8k = benchmark_gsm8k.evaluate(model=phi4)
# benchmark_gsm8k.predictions.to_csv("./gsm8k-smllama31-eval.csv")
# pd.DataFrame({'Metric': ["result"], 'Value': [results_gsm8k]}).to_csv('./gsm8k-llama31-result.csv', index=False)
#
# model = AutoModelForCausalLM.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/gemma-2-9b-it")
# tokenizer = AutoTokenizer.from_pretrained("/ukp-storage-1/zadorin/spt_llms/weights/gemma-2-9b-it")
#
# phi4 = CustomLM(model=model, tokenizer=tokenizer)
#
# benchmark_gsm8k = GSM8K(
#     n_problems=1319,
#     n_shots=8,
#     enable_cot=True
# )
#
# results_gsm8k = benchmark_gsm8k.evaluate(model=phi4)
# benchmark_gsm8k.predictions.to_csv("./gsm8k-gemma2-eval.csv")
# pd.DataFrame({'Metric': ["result"], 'Value': [results_gsm8k]}).to_csv('./gsm8k-gemma2-result.csv', index=False)
