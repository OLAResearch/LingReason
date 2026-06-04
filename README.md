# LingReason

LingReason contains code for generating structured linguistic reasoning traces from Universal Dependencies treebanks, dictionary glosses, and grammar rules. These traces are used for low-resource machine translation experiments, including in-context prompting, supervised fine-tuning, and reinforcement fine-tuning.

This repository accompanies our paper: [Reasoning over Grammar: Can Synthetic Linguistic Reasoning Traces Enhance Low-Resource Machine Translation?](https://arxiv.org/abs/2606.03782)

This public release only include Chintang (`ctn`) and Classical Armenian (`xcl`). The resource files (and relevant code) for Xibe (`xcl`) are excluded due to copyright restrictions.

## Example Data

The `data/` directory contains generated Chintang data as examples (`xcl` data not included due to file size limit). These files were produced with the code in `lingreason/` and can be used directly with the training, inference, and evaluation scripts in `scripts/`.

| File | Description |
| --- | --- |
| `data/ctn_test_icl.json` | Test set with linguistic reasoning guides (with placeholders) in the prompt, used for in-context learning experiment. |
| `data/ctn_train.json` | SFT/RFT train set with completed linguistic reasoning traces in the <think> block. |
| `data/ctn_eval.json` | Evaluation/Validation set with completed linguistic reasoning traces in the <think> block. |
| `data/ctn_no_thinking_train.json` | SFT Train set without linguistic reasoning traces. |
| `data/ctn_no_thinking_eval.json` | Evaluation/Validation set without linguistic reasoning traces. |
| `data/ctn_test.json` | Test set with completed linguistic reasoning traces in the <think> block, used for ICL baseline as well as SFT and RFT experiments. |

## Repository Structure

Reusable project code lives in `lingreason/`. These modules are primarily intended to be imported from Python code or notebooks.

| File | Role |
| --- | --- |
| `lingreason/generate_dict_from_ud_gloss.py` | Build lemma-to-gloss dictionaries from UD `.conllu` files. |
| `lingreason/generate_reasoning_traces_from_UD.py` | Generate reasoning-trace scaffolds from UD trees and grammar rules. |
| `lingreason/llm_fill_in_place_holders_api.py` | Fill reasoning-trace placeholders with the Gemini API. |
| `lingreason/generate_sft_data.py` | Build SFT datasets from completed traces. |
| `lingreason/generate_icl_eval_data.py` | Build ICL prompts with reasoning guides. |

Command-line entry points live in `scripts/`.

| File | Role |
| --- | --- |
| `scripts/sft.py` | Supervised fine-tuning script. |
| `scripts/rft-vllm-colocate.py` | GRPO/RFT training script with vLLM support. |
| `scripts/test_inference.py` | Model inference script for base or fine-tuned models. |
| `scripts/evaluate.py` | Evaluation entry point for BLEU, chrF, and SBERT-based scores. |

## Basic Usage Example

Supervised fine-tuning:

```bash
torchrun --standalone --nproc-per-node 8 scripts/sft.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --dataset-path data/ctn_train.json \
  --output-path outputs/sft \
  --max-steps 400
```

Reinforcement fine-tuning:

```bash
python -m torch.distributed.run --standalone --nproc-per-node 8 scripts/rft-vllm-colocate.py \
  --model path/to/sft_checkpoint_or_base_model \
  --train-files data/ctn_train.json \
  --eval-files data/ctn_eval.json \
  --output-dir outputs/rft \
  --max-steps 400
```

Inference:

```bash
python scripts/test_inference.py \
  --base-model Qwen/Qwen3-4B-Thinking-2507 \
  --test-file data/ctn_test.json \
  --output-file ctn_test_results.json
```

Evaluation:

```bash
python scripts/evaluate.py path/to/result_folder
```

To build new data, import the helper modules from `lingreason/`, for example:

```python
from lingreason.generate_reasoning_traces_from_UD import load_ud_trees
from lingreason.generate_icl_eval_data import generate_icl_eval_examples
from lingreason.generate_sft_data import generate_sft_examples
```

## Resource Files

Dictionary JSON files generated from UD glosses are in `dicts_generated_from_UD`.

You can generate dictionary files from appropriate UD `.conllu` folders using:

```bash
python -m lingreason.generate_dict_from_ud_gloss path/to/conllu_folder --output dicts_generated_from_UD/dict_UD_gloss_ctn.json
```

`gram_rules/` contains modular grammar rules paired with trigger conditions used by the trace-generation code. `gram_sketches/` contains stand-alone grammatical outline texts.

