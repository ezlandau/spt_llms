import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.schema import MultipleChoiceSchema
from deepeval.models.base_model import DeepEvalBaseLLM

SYSTEM_PROMPTS = {
    "llama31-8b": "Answer with a single letter: A, B, C, or D. Do not include any punctuation or extra text."
}

class CustomLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model_id: str,
        tokenizer_id: str,
        model_name: str,
        device: str = "cuda:0",
    ):
        self.device     = device
        self.model_name = model_name

        tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False)
        if isinstance(tok, bool):
            tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
        self.tokenizer = tok

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
        )
        model.resize_token_embeddings(len(tok))
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str, *_, **__) -> MultipleChoiceSchema:
        lm  = self.load_model()
        dev = next(lm.parameters()).device

        sys_p = SYSTEM_PROMPTS.get(self.model_name, "")
        full  = sys_p + "\n\n" + prompt + "\nAnswer:"

        inputs = self.tokenizer(full, return_tensors="pt").to(dev)

        outputs = lm.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            top_p=None,
            temperature=None,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        new_id     = outputs[0, prompt_len].item()

        raw = self.tokenizer.decode([new_id], skip_special_tokens=True).strip().upper()
        return MultipleChoiceSchema(answer=raw)

    async def a_generate(self, prompt: str) -> MultipleChoiceSchema:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


if __name__ == "__main__":
    llama31 = CustomLM(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_id="meta-llama/Llama-3.1-8B-Instruct",
        model_name="llama31-8b",
        device="cuda:0",
    )

    benchmark = MMLU(
        n_shots=5
    )
    result = benchmark.evaluate(model=llama31)

    benchmark.predictions.to_csv("./mmlu-llama31-8b-eval.csv", index=False)
    pd.DataFrame({"Metric": ["MMLU"], "Value": [result]}).to_csv(
        "./mmlu-llama31-8b-result.csv", index=False
    )
