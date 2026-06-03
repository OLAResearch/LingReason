#!/usr/bin/env python
# coding: utf-8

"""
GRPO training script for the vLLM-based LUMI path.

1. Start from an SFT checkpoint or base model.
2. Use GRPO to improve the final translation quality.
3. Keep a smaller reward on output format.
4. Add a process reward on the subset of examples that also contain gold
   reasoning traces. The process reward compares bracketed lexical/phrasal
   translations inside the <think> block, e.g. [the dark night].

Expected dataset fields
-----------------------
Each JSON example should contain:
  - prompt
  - answer

The gold "answer" field is expected to use the same tagged format as the
training target, so the script extracts:
  - gold_translation from <answer>...</answer>
  - gold_trace from <think>...</think>

Recommended launch pattern on LUMI
----------------------------------
Run the colocated vLLM launcher:
   sbatch rft-vllm.sh \
       --model /path/to/sft_checkpoint_or_model \
       --train-files data/train_trace.json data/train_parallel.json \
       --eval-files data/dev_trace.json data/dev_parallel.json \
       --output-dir /scratch/.../grpo_mt_run \
       --use-vllm \
       --use-peft

Notes
-----
- This script intentionally keeps the reward logic transparent and heavily
  commented so it is easy to modify.
- The script uses sentence-level rewards during RL and should be paired with
  your existing corpus-level evaluation script for final benchmarking.
- This vLLM version is configured for colocate mode in the current workflow.
"""

from __future__ import annotations

import os
import sys
import argparse
import inspect
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# # On LUMI, the container's /opt/venv site-packages can appear before the external venv on sys.path
# # Prepend the external venv's site-packages here so later imports (notably TRL)
# # consistently come from the intended venv rather than the container copy.
_venv = os.environ.get("VENV_PATH") or os.environ.get("VIRTUAL_ENV")
if _venv:
    for _site_packages in [
        os.path.join(_venv, "lib", "python3.12", "site-packages"),
        os.path.join(_venv, "lib64", "python3.12", "site-packages"),
    ]:
        if os.path.isdir(_site_packages):
            try:
                sys.path.remove(_site_packages)
            except ValueError:
                pass
            sys.path.insert(0, _site_packages)

import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel
from sacrebleu.metrics import BLEU, CHRF
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import wandb

import trl, vllm, sacrebleu, peft, transformers


# Regex helpers for checking exact output format used in your project.
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
STEP_RE = re.compile(r"(?im)^\s*Step\s+(\d+)\s*:")


DEBUG_PRINTED_GENERATION = False

def normalize_text(text: str) -> str:
    """
    Light normalization used before metric computation.
    We keep this conservative so we do not erase real distinctions.
    """
    text = (text or "").strip().lower() #replace None with "", strip leading/trailing whitespace, lowercase.
    text = re.sub(r"\s+", " ", text) #collapse repeated whitespace into a single space.
    return text


def tokenize_for_length(text: str) -> list[str]:
    """A tiny tokenizer used only for deciding whether a phrase is too short."""
    return [tok for tok in re.split(r"\s+", normalize_text(text)) if tok]


def extract_answer(response_text: str) -> str:
    """
    Extract the LAST <answer>...</answer> block.
    This mirrors the logic already used in test_inference.py.
    """
    matches = ANSWER_RE.findall(response_text or "")
    if matches:
        return matches[-1].strip()
    return ""


def extract_last_think_block(response_text: str) -> str:
    """Extract the LAST <think>...</think> block, or return an empty string."""
    matches = THINK_RE.findall(response_text or "")
    if matches:
        return matches[-1].strip()
    return ""


def extract_top_level_bracket_phrases(text: str) -> list[str]:
    """
    Extract top-level bracketed phrases while preserving nested brackets inside
    each phrase.

    Example:
    [2[SG].S-go[.SUBJ.NPST]] -> ["2[SG].S-go[.SUBJ.NPST]"]
    """
    phrases = []
    depth = 0
    start = None

    for idx, ch in enumerate(text or ""):
        if ch == "[":
            if depth == 0:
                start = idx + 1
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                phrase = text[start:idx].strip()
                if phrase:
                    phrases.append(phrase)
                start = None

    return phrases


def extract_bracket_phrases_from_think(response_text: str) -> list[str]:
    """Extract all bracketed phrases from the <think> block."""
    think_text = extract_last_think_block(response_text)
    return extract_top_level_bracket_phrases(think_text)

def load_json_datasets(paths: Iterable[str]) -> Dataset:
    """Load one or more JSON files and concatenate them into one Dataset."""
    datasets = [load_dataset("json", data_files=path, split="train") for path in paths]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets) # if more than one dataset, concatenate them


def build_default_run_name(model_name: str, train_files: Iterable[str]) -> str:
    """
    Build a default run name similar to sft.py:
    model_name#dataset_name

    If multiple train files are used, join their basenames with '+'.
    """
    model_path = Path(model_name)
    if model_path.is_absolute() and len(model_path.parts) >= 2:
        safe_model_name = "_".join(model_path.parts[-2:])
    else:
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    dataset_names = [os.path.splitext(os.path.basename(path))[0] for path in train_files]
    dataset_name = "+".join(dataset_names)
    return f"{safe_model_name}#RFT#{dataset_name}"


def build_default_output_dir(output_root: str, model_name: str, train_files: Iterable[str]) -> str:
    """
    Build a default output directory under the provided root, similar to sft.py.

    Example:
    /scratch/.../hf-data/Qwen_Qwen3-4B-Thinking-2507/RFT@ctn_train
    """
    model_path = Path(model_name)
    if model_path.is_absolute() and len(model_path.parts) >= 2:
        safe_model_name = "_".join(model_path.parts[-2:])
    else:
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    dataset_names = [os.path.splitext(os.path.basename(path))[0] for path in train_files]
    dataset_name = "+".join(dataset_names)
    return os.path.join(output_root, safe_model_name, f"RFT@{dataset_name}")


def build_prompt_formatter(tokenizer, args: argparse.Namespace):
    """Build the prompt formatter used by SFT, GRPO rollouts, and inference."""
    def _format_prompt(prompt_text: str) -> str:
        if args.prompt_template == "simple_wrapper":
            return f"### Prompt:\n{prompt_text}\n\n### Answer:\n"

        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )

    return _format_prompt


def prepare_dataset(
    dataset: Dataset,
    answer_field: str,
    format_prompt_fn,
) -> Dataset:
    """
    Add the exact columns the GRPO reward functions need.
    We keep all original columns too, because TRL can forward them to reward
    functions when remove_unused_columns=False.
    """

    def _map_example(example: dict) -> dict:
        prompt_text = example.get("prompt") or ""
        answer_text = example.get(answer_field) or ""
        gold_translation = extract_answer(answer_text)
        gold_trace = extract_last_think_block(answer_text)

        return {
            "prompt": format_prompt_fn(prompt_text),
            "gold_translation": gold_translation,
            "gold_trace": gold_trace,
        }

    dataset = dataset.map(
        _map_example,
        load_from_cache_file=False, # To avoid multi-process race condition, disable HF datasets disk caching for tiny preprocessing steps and keep them in memory.
        keep_in_memory=True, 
        )
    dataset = dataset.filter(
        lambda ex: bool((ex.get("prompt") or "").strip())
        and bool((ex.get(answer_field) or "").strip())
        and bool((ex.get("gold_translation") or "").strip()),
        load_from_cache_file=False, # To avoid multi-process race condition, disable HF datasets disk caching for tiny preprocessing steps and keep them in memory.
        keep_in_memory=True,
    )
    return dataset


@dataclass
class RewardSettings:
    """
    Hyperparameters for reward weights, all in one place
    """

    # Final-translation reward weights.
    final_chrf_weight: float = 0.55
    final_bleu_weight: float = 0.15
    final_sbert_weight: float = 0.25
    final_exact_bonus: float = 0.05
    empty_answer_penalty: float = 0.25

    # Format reward components.
    think_tag_bonus: float = 0.10
    answer_tag_bonus: float = 0.10
    correct_order_bonus: float = 0.05
    non_empty_answer_bonus: float = 0.10
    has_step_bonus: float = 0.10
    starts_at_step_one_bonus: float = 0.03
    monotonic_step_bonus: float = 0.02
    trailing_text_penalty: float = 0.05
    missing_think_penalty: float = 0.10
    missing_answer_penalty: float = 0.20
    empty_tag_penalty: float = 0.20
    malformed_step_penalty: float = 0.10
    wrong_order_penalty: float = 0.10

    # Process reward on bracketed phrases.
    short_phrase_exact_weight: float = 0.65
    short_phrase_chrf_weight: float = 0.35
    long_phrase_exact_weight: float = 0.15
    long_phrase_chrf_weight: float = 0.70
    long_phrase_sbert_weight: float = 0.15
    process_recall_weight: float = 0.75
    process_precision_weight: float = 0.20
    process_non_empty_bonus: float = 0.05
    process_missing_trace_penalty: float = 0.10


class MTRewardFunctions:
    """
    Bundle all reward functions into one object.

    Why use a class?
    - We want to share metric objects across reward calls.
    - We lazily load SBERT once instead of reloading it every step.
    - It keeps the top-level training code cleaner.
    """

    def __init__(self, sbert_model_name: str, sbert_device: str, settings: RewardSettings):
        self.settings = settings
        self.bleu = BLEU(lowercase=True, effective_order=True, tokenize="13a", smooth_method="exp")
        self.chrf = CHRF()
        self.sbert_model_name = sbert_model_name
        self.sbert_device = sbert_device
        self.sbert = None

    def _maybe_print_debug_generation(self, prompts, completions) -> None:
        """
        Print one example completion for manual inspection.

        This is intentionally lightweight:
        - only rank 0 prints
        - only the first observed example is printed
        - gated by an environment variable so the script logic stays simple
        """
        global DEBUG_PRINTED_GENERATION

        if DEBUG_PRINTED_GENERATION:
            return

        if os.environ.get("RANK", "0") != "0":
            return

        if os.environ.get("DEBUG_PRINT_GENERATIONS", "0") != "1":
            return

        if not completions:
            return

        prompt_text = prompts[0] if prompts else ""
        completion_text = completions[0] or ""

        print("=== debug generation sample ===")
        print("prompt:")
        print(prompt_text)
        print("completion:")
        print(completion_text)
        print("extracted_answer:")
        print(extract_answer(completion_text))
        print("=== end debug generation sample ===")

        DEBUG_PRINTED_GENERATION = True

    def _ensure_sbert(self) -> SentenceTransformer | None:
        """
        SBERT is optional. If loading fails, the script continues with the
        lexical metrics only instead of crashing the whole RL run.
        """
        if self.sbert is not None:
            return self.sbert

        if self.sbert_model_name.lower() in {"", "none", "off"}:
            return None

        try:
            self.sbert = SentenceTransformer(self.sbert_model_name, device=self.sbert_device)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"WARNING: failed to load SBERT model '{self.sbert_model_name}': {exc}")
            print("WARNING: continuing without SBERT-based reward components.")
            self.sbert = None

        return self.sbert

    def _semantic_similarity(self, prediction: str, reference: str) -> float:
        """
        Sentence-level semantic similarity in [0, 1].
        We shift cosine similarity from [-1, 1] into [0, 1] for convenience.
        """
        model = self._ensure_sbert()
        if model is None:
            return 0.0

        pred_emb = model.encode([prediction], convert_to_tensor=True)
        ref_emb = model.encode([reference], convert_to_tensor=True)
        score = F.cosine_similarity(pred_emb, ref_emb).item()
        return max(0.0, min(1.0, (score + 1.0) / 2.0))

    def _translation_similarity(self, prediction: str, reference: str) -> float:
        """
        Reward for the final translation.
        This is the main task reward and should dominate the total signal.
        """
        prediction_norm = normalize_text(prediction)
        reference_norm = normalize_text(reference)

        if not prediction_norm:
            return -self.settings.empty_answer_penalty

        bleu_score = self.bleu.sentence_score(prediction_norm, [reference_norm]).score / 100.0
        chrf_score = self.chrf.sentence_score(prediction_norm, [reference_norm]).score / 100.0
        sbert_score = self._semantic_similarity(prediction_norm, reference_norm)
        exact_bonus = self.settings.final_exact_bonus if prediction_norm == reference_norm else 0.0

        return (
            self.settings.final_bleu_weight * bleu_score
            + self.settings.final_chrf_weight * chrf_score
            + self.settings.final_sbert_weight * sbert_score
            + exact_bonus
        )

    def _phrase_similarity(self, prediction: str, reference: str) -> float:
        """
        Similarity for bracketed lexical/phrasal translations.
        Short phrases are mostly lexical and benefit from exact-match pressure.
        Longer phrases can use a small semantic component.
        """
        prediction_norm = normalize_text(prediction)
        reference_norm = normalize_text(reference)

        if not prediction_norm or not reference_norm:
            return 0.0

        exact = 1.0 if prediction_norm == reference_norm else 0.0
        chrf_score = self.chrf.sentence_score(prediction_norm, [reference_norm]).score / 100.0
        max_len = max(len(tokenize_for_length(prediction_norm)), len(tokenize_for_length(reference_norm)))

        if max_len <= 2:
            return (
                self.settings.short_phrase_exact_weight * exact
                + self.settings.short_phrase_chrf_weight * chrf_score
            )

        sbert_score = self._semantic_similarity(prediction_norm, reference_norm)
        return (
            self.settings.long_phrase_exact_weight * exact
            + self.settings.long_phrase_chrf_weight * chrf_score
            + self.settings.long_phrase_sbert_weight * sbert_score
        )

    def _aggregate_phrase_reward(self, gold_phrases: list[str], pred_phrases: list[str]) -> float:
        """
        Compare the gold and predicted bracketed phrases.

        We do not require exact one-to-one alignment by position, because the
        model may phrase or order parts slightly differently. Instead:
        - recall side: for each gold phrase, find the best generated phrase
        - precision side: for each generated phrase, find the best gold phrase

        This gives a soft set/sequence comparison that is robust enough for RL.
        """
        if not gold_phrases:
            return 0.0

        if not pred_phrases:
            return -self.settings.process_missing_trace_penalty
        #For each gold phrase, find the best-matching predicted phrase.
        recall_scores = [
            max(self._phrase_similarity(pred_phrase, gold_phrase) for pred_phrase in pred_phrases)
            for gold_phrase in gold_phrases
        ]
        #For each predicted phrase, find the best-matching gold phrase.
        precision_scores = [
            max(self._phrase_similarity(pred_phrase, gold_phrase) for gold_phrase in gold_phrases)
            for pred_phrase in pred_phrases
        ]

        recall = sum(recall_scores) / len(recall_scores)
        precision = sum(precision_scores) / len(precision_scores)
        # the default is intentionally recall-heavy, covering the gold partial translations matters most
        reward = (
            self.settings.process_recall_weight * recall
            + self.settings.process_precision_weight * precision
            + self.settings.process_non_empty_bonus
        )
        return reward

    def translation_reward_func(self, completions, gold_translation, **kwargs):
        """Reward based only on the extracted final answer."""
        self._maybe_print_debug_generation(kwargs.get("prompts"), completions)
        rewards = []
        for completion, reference in zip(completions, gold_translation):
            predicted_answer = extract_answer(completion)
            rewards.append(self._translation_similarity(predicted_answer, reference))
        return rewards

    def format_reward_func(self, completions, **kwargs):
        """
        Reward the required tagged output format:
        <think> ... Step 1: ... </think>
        <answer> ... </answer>

        This reward is intentionally structural, not semantic.
        """
        rewards = []

        for completion in completions:
            reward = 0.0

            think_matches = list(THINK_RE.finditer(completion))
            answer_matches = list(ANSWER_RE.finditer(completion))

            has_think = bool(think_matches)
            has_answer = bool(answer_matches)

            if has_think:
                reward += self.settings.think_tag_bonus
            else:
                reward -= self.settings.missing_think_penalty

            if has_answer:
                reward += self.settings.answer_tag_bonus
            else:
                reward -= self.settings.missing_answer_penalty

            if has_think and has_answer:
                last_think = think_matches[-1]
                last_answer = answer_matches[-1]
                think_text = last_think.group(1).strip()
                answer_text = last_answer.group(1).strip()

                if last_think.start() < last_answer.start(): #reward correct order <think> before <answer>
                    reward += self.settings.correct_order_bonus
                else:
                    reward -= self.settings.wrong_order_penalty

                if answer_text:#reward non-empty answer text
                    reward += self.settings.non_empty_answer_bonus
                else:
                    reward -= self.settings.empty_tag_penalty

                step_numbers = [int(m.group(1)) for m in STEP_RE.finditer(think_text)]
                if step_numbers:
                    reward += self.settings.has_step_bonus
                    if step_numbers[0] == 1:
                        reward += self.settings.starts_at_step_one_bonus
                    if step_numbers == sorted(step_numbers) and len(step_numbers) == len(set(step_numbers)):
                        reward += self.settings.monotonic_step_bonus
                else:
                    reward -= self.settings.malformed_step_penalty

                trailing_text = completion[last_answer.end():].strip() # penalize text after final </answer>
                if trailing_text:
                    reward -= self.settings.trailing_text_penalty

            rewards.append(reward)

        return rewards

    def process_reward_func(self, completions, gold_trace, **kwargs):
        """
        Important:
        - This function assumes every training example has a gold trace. Which need to be changed, because 3500 parallel examples do not have gold traces.
        Process reward based on bracketed lexical/phrasal translations inside
        the <think> block.
        """
        rewards = []

        for completion, trace in zip(completions, gold_trace):
            # extract bracketed phrases from the generated completion and the gold reasoning trace
            predicted_phrases = extract_bracket_phrases_from_think(completion)
            gold_phrases = extract_top_level_bracket_phrases(trace or "")

            rewards.append(self._aggregate_phrase_reward(gold_phrases, predicted_phrases))

        return rewards


def parse_args() -> argparse.Namespace:
    """Command-line interface for the GRPO training job."""
    parser = argparse.ArgumentParser(description="Train a machine-translation GRPO model with format and process rewards.")

    # Core model/data arguments.
    parser.add_argument("--model", type=str, required=True, help="Base model or SFT checkpoint to continue training from.")
    parser.add_argument(
        "--sft-lora-path",
        type=str,
        default=None,
        help="Optional path to a previously saved SFT LoRA checkpoint to continue training from.",
    )
    parser.add_argument("--train-files", type=str, nargs="+", required=True, help="One or more JSON files for GRPO training.")
    parser.add_argument("--eval-files", type=str, nargs="*", default=None, help="Optional JSON files for GRPO evaluation.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where checkpoints and logs will be written.")

    # Dataset schema argument. Gold translation and trace are both extracted
    # directly from the tagged answer field.
    parser.add_argument("--answer-field", type=str, default="answer", help="Field containing the gold tagged answer.")

    # General training arguments.
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=7)
    parser.add_argument("--report-to", type=str, default="wandb", help="Trainer reporting backend, e.g. wandb or none.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional experiment name for logging.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Optional Trainer checkpoint path to resume from.")
    parser.add_argument(
        "--debug-print-generations",
        action="store_true",
        help="Print one sampled completion on rank 0 for quick manual inspection.",
    )
    parser.add_argument(
        "--package-diagnostic",
        action="store_true",
        help="Print runtime package/import diagnostics on rank 0 before training starts.",
    )

    # Generation / GRPO-specific arguments.
    parser.add_argument("--num-generations", type=int, default=4, help="How many candidate completions to sample per prompt.")
    parser.add_argument("--num-generations-eval", type=int, default=2, help="How many completions to sample per prompt at eval time.")
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--scale-rewards",
        type=str,
        default="batch",
        choices=["group", "batch", "none"],
        help="Reward-scaling strategy. 'batch' is a good default for this setup.",
    )
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient. 0.0 keeps memory lower and matches current TRL defaults.")

    # vLLM integration options.
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM generation through TRL.")
    parser.add_argument("--vllm-mode", type=str, default="colocate", choices=["server", "colocate"])
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-max-model-length", type=int, default=None)
    parser.add_argument("--vllm-model-impl", type=str, default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--vllm-enable-sleep-mode", action="store_true", help="Offload vLLM weights/cache during optimizer steps to reduce colocated GPU memory pressure.")

    # Tokenizer/model loading.
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Load/train in bfloat16 when supported.")
    parser.add_argument("--prompt-template", type=str, default="simple_wrapper", choices=["simple_wrapper", "qwen_chat"], help="Prompt formatting to use.")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode for qwen_chat prompting.")

    # PEFT / LoRA options.
    parser.add_argument("--use-peft", action="store_true", help="Wrap the model with LoRA for cheaper RL fine-tuning.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="all-linear", help='Use "all-linear" or a comma-separated module list.')

    # Reward weighting between the reward functions passed to GRPO.
    parser.add_argument("--translation-reward-weight", type=float, default=0.75)
    parser.add_argument("--format-reward-weight", type=float, default=0.10)
    parser.add_argument("--process-reward-weight", type=float, default=0.15)
    parser.add_argument(
        "--disable-process-reward",
        action="store_true",
        help="Disable the process reward entirely, even if process-reward-weight is non-zero.",
    )

    # SBERT reward options.
    parser.add_argument("--sbert-model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--sbert-device",
        type=str,
        default="cpu",
        help='Device used by SentenceTransformer. "cpu" is the safest default for RL stability.',
    )

    return parser.parse_args()


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    """Create the LoRA config used by GRPOTrainer when --use-peft is enabled."""
    target_modules = args.lora_target_modules
    if target_modules != "all-linear":
        target_modules = [module.strip() for module in target_modules.split(",") if module.strip()]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        exclude_modules=r".*visual\..*", # exclude visual modules for models like Qwen3.5
        task_type="CAUSAL_LM",
    )


def log_vllm_model_identity(model, args: argparse.Namespace) -> None:
    """
    Print the model identity that TRL/vLLM colocate is likely to use.

    Why this matters:
    - In colocate mode, upstream TRL has been discussed as constructing the
      internal vLLM engine from `model.name_or_path`.
    - When `model` is a PEFT-wrapped model, that value may still point to the
      base model identifier rather than the LoRA adapter checkpoint.
    - These prints let us verify, from the training logs, what identity is
      being forwarded and whether the active model is a PeftModel.
    """
    if not args.use_vllm:
        return

    print("=== vLLM model identity diagnostic ===")
    print(f"use_vllm={args.use_vllm}")
    print(f"vllm_mode={args.vllm_mode}")
    print(f"python_model_type={type(model)}")
    print(f"model.name_or_path={getattr(model, 'name_or_path', None)}")

    if isinstance(model, PeftModel):
        print("model_is_peft=True")
        print(f"active_adapter={getattr(model, 'active_adapter', None)}")
        print(f"peft_adapter_path={args.sft_lora_path}")

        base_model = getattr(model, "base_model", None)
        if base_model is not None:
            print(f"peft_base_model_type={type(base_model)}")
            print(f"peft_base_model.name_or_path={getattr(base_model, 'name_or_path', None)}")

            wrapped_base_model = getattr(base_model, "model", None)
            if wrapped_base_model is not None:
                print(f"peft_base_model.model_type={type(wrapped_base_model)}")
                print(
                    "(The model being fine-tuned:)peft_base_model.model.name_or_path="
                    f"{getattr(wrapped_base_model, 'name_or_path', None)}"
                )
    else:
        print("model_is_peft=False")

    print(
        "If TRL colocate initializes vLLM from model.name_or_path, the value above is the "
        "key clue for whether generation starts from the base model id/path or the adapter path."
    )
    print("=== end vLLM model identity diagnostic ===")


def log_package_diagnostic(args: argparse.Namespace) -> None:
    """
    Print the active Python/package environment and probe the tensor-parallel
    import that PEFT hits during resume_from_checkpoint.
    """
    print("=== package diagnostic ===")
    print(f"python={sys.executable}")
    print(f"resume_from_checkpoint={args.resume_from_checkpoint}")
    print(f"sft_lora_path={args.sft_lora_path}")
    print("sys.path:")
    for path in sys.path:
        print(f"  {path}")

    packages = {
        "trl": trl,
        "vllm": vllm,
        "peft": peft,
        "transformers": transformers,
        "sacrebleu": sacrebleu,
    }
    for name, module in packages.items():
        print(f"{name}: version={getattr(module, '__version__', 'unknown')}")
        print(f"{name}: file={getattr(module, '__file__', 'unknown')}")

    # try:
    #     tp_module = importlib.import_module("transformers.integrations.tensor_parallel")
    #     print(f"tensor_parallel module={getattr(tp_module, '__file__', 'unknown')}")
    #     print(f"has EmbeddingParallel={hasattr(tp_module, 'EmbeddingParallel')}")
    #     try:
    #         from transformers.integrations.tensor_parallel import EmbeddingParallel
    #         print(f"EmbeddingParallel import=OK ({EmbeddingParallel})")
    #     except Exception as exc:
    #         print(f"EmbeddingParallel import FAILED: {exc!r}")
    # except Exception as exc:
    #     print(f"tensor_parallel module import FAILED: {exc!r}")

    print("=== end package diagnostic ===")


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))

    if args.debug_print_generations:
        os.environ["DEBUG_PRINT_GENERATIONS"] = "1"

    if args.package_diagnostic and rank == 0:
        log_package_diagnostic(args)

    if args.run_name is None:
        args.run_name = build_default_run_name(args.model, args.train_files)

    args.output_dir = build_default_output_dir(args.output_dir, args.model, args.train_files)
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load tokenizer. GRPO requires left padding and a pad token.
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    format_prompt_fn = build_prompt_formatter(tokenizer, args)

    # -----------------------------------------------------------------------
    # 1b. Load the training model.
    #     Two modes are supported:
    #     - Base model only, optionally adding a fresh LoRA adapter via --use-peft
    #     - Base model + previously saved SFT LoRA checkpoint via --sft-lora-path
    # -----------------------------------------------------------------------
    torch_dtype = torch.bfloat16 if args.bf16 else None
    model_init_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        **({"torch_dtype": torch_dtype} if torch_dtype is not None else {}),
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_init_kwargs,
    )

    if args.sft_lora_path:
        # Continue RL from an already trained LoRA adapter.
        # is_trainable=True keeps the adapter weights trainable during GRPO.
        model = PeftModel.from_pretrained(
            model,
            args.sft_lora_path,
            is_trainable=True,
        )
        print(f"Loaded SFT LoRA checkpoint from: {args.sft_lora_path}")

    model.enable_input_require_grads()
    model.config.use_cache = False

    # Diagnostic for the PEFT + vLLM colocate question:
    # print the model identity before GRPOTrainer creates its internal vLLM
    # engine so the logs show what TRL is likely forwarding to vLLM.
    if rank == 0:
        log_vllm_model_identity(model, args)

    # -----------------------------------------------------------------------
    # 2. Load and normalize datasets.
    # -----------------------------------------------------------------------
    train_dataset = load_json_datasets(args.train_files)
    train_dataset = prepare_dataset(
        train_dataset,
        answer_field=args.answer_field,
        format_prompt_fn=format_prompt_fn,
    )

    eval_dataset = None
    if args.eval_files:
        eval_dataset = load_json_datasets(args.eval_files)
        eval_dataset = prepare_dataset(
            eval_dataset,
            answer_field=args.answer_field,
            format_prompt_fn=format_prompt_fn,
        )

    print(f"Prepared {len(train_dataset)} training examples.")
    if eval_dataset is not None:
        print(f"Prepared {len(eval_dataset)} evaluation examples.")
    print(f"Run name: {args.run_name}")

    # In distributed training, only rank 0 should create and write to the W&B
    # run. Otherwise torchrun spawns one W&B run per process.
    effective_report_to = args.report_to
    if rank != 0 and args.report_to.lower() != "none":
        effective_report_to = "none"

    # -----------------------------------------------------------------------
    # 3. Build reward functions.
    #    We pass three reward functions to TRL and combine them with
    #    reward_weights:
    #      - final translation reward
    #      - output-format reward
    #      - process reward on trace examples only
    # -----------------------------------------------------------------------
    reward_settings = RewardSettings()
    rewards = MTRewardFunctions(
        sbert_model_name=args.sbert_model_name,
        sbert_device=args.sbert_device,
        settings=reward_settings,
    )

    reward_funcs = []
    reward_weights = []
    active_reward_names = []

    # Final translation reward is the main task objective, so it is kept
    # whenever its weight is non-zero.
    if args.translation_reward_weight != 0.0:
        reward_funcs.append(rewards.translation_reward_func)
        reward_weights.append(args.translation_reward_weight)
        active_reward_names.append("translation")

    # Format reward is optional. Setting its weight to 0.0 cleanly removes it.
    if args.format_reward_weight != 0.0:
        reward_funcs.append(rewards.format_reward_func)
        reward_weights.append(args.format_reward_weight)
        active_reward_names.append("format")

    # Process reward can be disabled explicitly with a flag, or implicitly by
    # setting its weight to 0.0.
    if not args.disable_process_reward and args.process_reward_weight != 0.0:
        reward_funcs.append(rewards.process_reward_func)
        reward_weights.append(args.process_reward_weight)
        active_reward_names.append("process")

    if not reward_funcs:
        raise ValueError(
            "No active reward functions remain. Keep translation reward enabled, "
            "or set at least one reward weight to a non-zero value."
        )

    print(f"Active rewards: {', '.join(active_reward_names)}")
    print(f"Reward weights: {reward_weights}")

    # -----------------------------------------------------------------------
    # 4. Create the GRPO training config.
    #    remove_unused_columns=False is important because the reward functions
    #    need dataset columns like gold_translation and gold_trace.
    # -----------------------------------------------------------------------
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        bf16=args.bf16,
        report_to="none" if effective_report_to.lower() == "none" else effective_report_to,
        run_name=args.run_name,
        seed=args.seed,
        remove_unused_columns=False,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        reward_weights=reward_weights,
        scale_rewards=args.scale_rewards,
        beta=args.beta,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_model_length=args.vllm_max_model_length,
        vllm_model_impl=args.vllm_model_impl,
        log_completions=True,
        log_unique_prompts=False,
        vllm_importance_sampling_correction=False, # not applying importance sampling correction to help stabilize training and avoid failure where sampling_per_token_logps_list sometimes contains None
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
    )

    # -----------------------------------------------------------------------
    # 5. Optional LoRA wrapping.
    #    For your use case this is usually the best starting point because RL
    #    can otherwise become very memory-hungry.
    # -----------------------------------------------------------------------
    peft_config = None
    if args.sft_lora_path:
        if args.use_peft:
            print("Ignoring --use-peft because --sft-lora-path already provides a trainable LoRA adapter.")
    elif args.use_peft:
        peft_config = build_lora_config(args)

    # -----------------------------------------------------------------------
    # 6. Build the GRPO trainer.
    # -----------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # -----------------------------------------------------------------------
    # 7. Train and save.
    # -----------------------------------------------------------------------
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)

    print(f"Training complete. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
