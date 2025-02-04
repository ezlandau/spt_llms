import os
import pandas as pd
import gc
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from utils.refusal_eval import refusal_eval
from utils.safety_eval import safety_eval

# Set Triton cache directory
os.environ["TRITON_CACHE_DIR"] = "/ukp-storage-1/zadorin/spt_llms/triton_cache"
os.makedirs("/ukp-storage-1/zadorin/spt_llms/triton_cache", exist_ok=True)

device = "cuda"
print(f"Using device: {device}")

models = {
    "gemma2-9b": "/storage/ukp/shared/shared_model_weights/models--google--gemma-2-9b-it",
    "phi-4-14b": "/ukp-storage-1/zadorin/spt_llms/weights/phi-4",
    "llama31-8b": "/storage/ukp/shared/shared_model_weights/models--llama-3/8B-Instruct" # maybe change to just Llama3
}

benchmarks = {
    "advbench": {
        "input": "./data/ab.csv",
        "completion": "./completions/ab-{model}.csv",
        "evaluation": "./evaluations/ab-{model}-eval.csv",
        "prompt_col": "prompt"
    },
    "or-bench-toxic": {
        "input": "./data/obt.csv",
        "completion": "./completions/obt-{model}.csv",
        "evaluation": "./evaluations/obt-{model}-eval.csv",
        "prompt_col": "prompt"
    },
    "xstest-unsafe": {
        "input": "./data/xsu.csv",
        "completion": "./completions/xsu-{model}.csv",
        "evaluation": "./evaluations/xsu-{model}-eval.csv",
        "prompt_col": "prompt"
    },
    "do-not-answer": {
        "input": "./data/dna.csv",
        "completion": "./completions/dna-{model}.csv",
        "evaluation": "./evaluations/dna-{model}-eval.csv",
        "prompt_col": "question"
    },
    "strong-reject": {
        "input": "./data/sr.csv",
        "completion": "./completions/sr-{model}.csv",
        "evaluation": "./evaluations/sr-{model}-eval.csv",
        "prompt_col": "forbidden_prompt"
    },
    # "or-bench-hard-1k": {
    #     "input": "./data/obh.csv",
    #     "completion": "./completions/obh-{model}.csv",
    #     "evaluation": "./evaluations/obh-{model}-eval.csv",
    #     "prompt_col": "prompt"
    # },
    # "xstest-safe": {
    #     "input": "./data/xss.csv",
    #     "completion": "./completions/xss-{model}.csv",
    #     "evaluation": "./evaluations/xss-{model}-eval.csv",
    #     "prompt_col": "prompt"
    # }
}

def generate_responses(llm, data, prompt_col):

    prompts = data[prompt_col].apply(
        lambda prompt: (
            "### User Question:\n" +
            prompt +
            "\n### Response:\n"
        )
    ).tolist()

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,  # Deterministic output
        top_p=1.0,
        top_k=1  # Greedy decoding
    )

    responses = llm.generate(prompts, sampling_params)

    generated_texts = [
        response.outputs[0].text.strip().split("### Response:")[-1].strip()
        for response in responses
    ]

    data["completion"] = generated_texts
    return data

def main():
    for (model_name, model_path) in models.items():
        max_model_len = 16384 if "phi-4-14b" in model_name else None
        llm = LLM(
            model_path,
            device=device,
            max_model_len=max_model_len,
        )
        for benchmark in benchmarks.keys():
            data = pd.read_csv(benchmarks[benchmark]["input"])
            print(f"Generating completions for model: {model_name} on the benchmark: {benchmark}")
            output_file = benchmarks[benchmark]["completion"].format(model=model_name)
            updated_data = generate_responses(llm, data, benchmarks[benchmark]["prompt_col"])
            updated_data.to_csv(output_file, index=False)
            print(f"Completions for {model_name} on benchmark: {benchmark} saved to {output_file}")

            completions_file = output_file
            output_file = benchmarks[benchmark]["evaluation"].format(model=model_name)
            print(f"Generating evaluations for model: {model_name} on the benchmark: {benchmark}")
            if benchmark == "or-bench-hard-1k" or benchmark == "xstest-safe":
                refusal_eval(completions_file, output_file)
            else:
                safety_eval(completions_file, output_file, benchmarks[benchmark]["prompt_col"])
            print(f"Evaluations for {model_name} on benchmark: {benchmark} saved to {output_file}")

        del llm.llm_engine.model_executor.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Successfully deleted the llm pipeline for {model_name} and freed the GPU memory")
    destroy_model_parallel()

#TODO Run performance eval with utils just for models
#TODO When DeepSeek API is online, complete refusal evals

if __name__ == "__main__":
    main()
