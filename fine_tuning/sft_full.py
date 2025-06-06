import os
import torch
import wandb

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_quantization_config,
)

TRITON_CACHE_DIR = "/ukp-storage-1/zadorin/spt_llms/triton_cache"
os.environ["TRITON_CACHE_DIR"] = TRITON_CACHE_DIR
os.makedirs(TRITON_CACHE_DIR, exist_ok=True)

MODEL_PATH   = "/ukp-storage-1/zadorin/spt_llms/weights/Llama-3.1-8B-Instruct"
MODEL_NAME   = MODEL_PATH.split("/")[-1]
DATASET_PATH = "/ukp-storage-1/zadorin/spt_llms/sensitive_words/sw_sft.jsonl"
PROJECT_NAME = "SFT-SW"
EXP_NAME     = f"{MODEL_NAME}-FULL_SW_ALL"

os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_ENTITY"]  = "RADFAN"
wandb.login(key="")

model_config = ModelConfig(
    model_name_or_path   = MODEL_PATH,
    use_peft             = False,
    load_in_8bit         = False,
    load_in_4bit         = False,
    bnb_4bit_quant_type  = "nf4",
    use_bnb_nested_quant = True,
    torch_dtype          = "bfloat16",
)

training_args = SFTConfig(
    dataset_num_proc            = 16,
    max_seq_length              = 1024,
    run_name                    = EXP_NAME,
    output_dir                  = f"/ukp-storage-1/zadorin/spt_llms/{PROJECT_NAME}/{EXP_NAME}",
    num_train_epochs            = 3,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 4,
    gradient_checkpointing      = False,
    bf16                        = True,
    learning_rate               = 3e-6,
    adam_epsilon                = 1e-5,
    logging_steps               = 20,
    eval_strategy               = "steps",
    eval_steps                  = 200,
    save_steps                  = 200,
    max_grad_norm               = 1.0,
    remove_unused_columns       = True,
    push_to_hub                 = False,
    hub_model_id                = f"ez-landau/{PROJECT_NAME}-{EXP_NAME}",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
)

initial_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    pad_token    = "<|pad|>",
    padding_side = "right",
)

quant_config = get_quantization_config(model_config)
torch_dtype  = getattr(torch, model_config.torch_dtype)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config = quant_config,
    torch_dtype         = torch_dtype,
)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model.config.pad_token_id = tokenizer.pad_token_id

raw = load_dataset("json", data_files=DATASET_PATH, split="train") \
    .map(lambda ex, idx: {"block": idx // 1020}, with_indices=True)

splits = {"train": [], "validation": []}
for b in range(4):
    block_ds = raw.filter(lambda ex: ex["block"] == b)
    part    = block_ds.train_test_split(test_size=0.2, seed=42, shuffle=True)
    splits["train"].append(part["train"])
    splits["validation"].append(part["test"])

train_dataset = concatenate_datasets(splits["train"]).shuffle(seed=42)
eval_dataset  = concatenate_datasets(splits["validation"]).shuffle(seed=42)

cols_to_remove = ["messages", "block"]
def preprocess_fn(batch):
    texts = []
    for msgs in batch["messages"]:
        conv = ""
        for m in msgs:
            prefix = "<|user|>" if m["role"]=="user" else "<|assistant|>"
            conv  += f"{prefix} {m['content']}\n"
        texts.append(conv)
    tok = tokenizer(
        texts,
        truncation=True,
        max_length=training_args.max_seq_length,
        padding="max_length",
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

train_dataset = train_dataset.map(preprocess_fn, batched=True, remove_columns=cols_to_remove)
eval_dataset  = eval_dataset.map(preprocess_fn, batched=True, remove_columns=cols_to_remove)

def main() -> None:
    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
    )
    trainer.train()

    # finalize tokenizer & embeddings
    trainer.model.resize_token_embeddings(len(tokenizer) - 1)
    trainer.tokenizer = initial_tokenizer

    try:
        trainer.push_to_hub()
        print("Model pushed to hub successfully.")
    except Exception as e:
        print(f"Push to hub failed: {e}")

if __name__ == "__main__":
    main()
