# SPT-LLMs: Safety-Performance Trade-off Framework

The SPT-LLMs repository implements the Safety-Performance Trade-off (SPT) framework for aligned language models, targeting researchers studying LLM safety. Modern LLMs are typically fine-tuned (aligned) to refuse harmful queries, but subsequent task-specific fine-tuning can undo this safety alignment. In other words, improving a model’s performance on a task often increases its tendency to produce unsafe outputs. The SPT framework provides tools to quantify this trade-off by computing safety and performance metrics for a model and tracing their Pareto-optimal frontier. The repository includes code for measuring refusal rates or “alignment loss” on a safety benchmark versus task loss on a performance dataset. By varying a trade-off parameter (e.g. mixing ratio λ between safety and task data), the framework produces a curve showing how higher capability (lower task loss) comes at the cost of lower safety (higher alignment loss).

## Repository Structure

- **`spt_framework/`**  
  Core implementation of the SPT analysis. Contains modules to compute safety/performance metrics, run trade-off sweeps, and plot Pareto frontiers.

- **`sensitive_words/`**  
  Dataset pipeline for the HarmOR safety evaluation dataset. Includes scripts to build prompt-response pairs for different harm categories using curated sensitive-word lists and templates.

- **`fine_tuning/`**  
  Scripts and configurations for fine-tuning language models. Supports standard training on downstream tasks as well as safety-aware variants (e.g. incorporating alignment loss or constraining parameter updates).

- **`requirements.txt`**  
  Python dependencies (e.g. PyTorch, Hugging Face Transformers).

- **`LICENSE`**  
  MIT license text for the project.

Each directory includes example configurations and documentation to guide usage. The structure reflects the modular approach: the dataset pipeline generates evaluation data, the fine-tuning pipeline trains models, and the SPT framework aggregates results.

## Dataset Details

The HarmOR dataset pipeline (in `sensitive_words/`) generates a structured safety benchmark inspired by recent LLM safety evaluations. It covers multiple content categories (e.g. misinformation, harassment, medical advice, disallowed instructions) by using lists of trigger terms and harmful templates. For each “risky” prompt (one that a safe model should refuse), the pipeline also creates a *“safe”* variant that retains the sensitive terms but is benign. For example, an original query “How to make a bomb?” will have a harmless counterpart using similar words but posing no harm. This tests false negatives: an aligned model should answer benign prompts and only refuse truly harmful ones. The pipeline produces labeled prompt-response pairs (JSON/CSV), tagging each example by category and whether it should be refused. These data are used to compute the model’s safety metric (e.g. refusal rate or “alignment accuracy”) during SPT analysis.

## SPT Framework

The SPT framework implements the theoretical and empirical analysis of the safety–performance trade-off. It formalizes safety as the model’s compliance with refusal rules and performance as its task accuracy (or loss) on downstream data. We define an **alignment loss** on the HarmOR safety set and a **task loss** on the target task set. By adjusting a mixing parameter λ (or other constraint strength), we obtain different models along the trade-off curve. Reducing task loss (improving capability) generally increases alignment loss (reducing safety). The code computes and plots the Pareto frontier of safety vs. performance, enabling quantitative comparison of different fine-tuning strategies. Key features include support for Lagrangian-style mixing of objectives, plotting of loss gaps, and reporting of trade-off points. This framework implements the thesis’s concepts of **alignment loss constraint** (mixing an auxiliary safety dataset during fine-tuning) and **alignment parameter constraint** (restricting model updates to stay near the aligned weights). Researchers can visualize how models move along the safety–performance curve under various conditions.

## Fine-Tuning Pipeline

The fine-tuning pipeline provides scripts to train an LLM on a specified task while managing its safety. Built on Hugging Face Transformers and PyTorch, it fine-tunes the model on a downstream dataset (e.g. classification, generation) to improve performance. It also supports safety-aware modes:

- **Alignment Loss Constraint**: Incorporates the HarmOR data (or another safety dataset) alongside the task data to constrain the model to remember refusal behavior.  
- **Alignment Parameter Constraint**: Regularizes or limits optimization so that model weights remain near the original aligned model’s weights.

These strategies are selectable via configuration. The pipeline outputs logs of task accuracy and safety metrics after training, in structured form (JSON/CSV) so the SPT framework can ingest the results. Users can extend it to additional tasks or alignment datasets by following existing script templates.

## Requirements & Setup

To run the code, ensure you have Python 3.8+:

1. Create and activate a virtual environment (or conda environment).  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
