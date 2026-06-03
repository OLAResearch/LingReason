import sys
import os

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

import os
import argparse
import json
import torch
import re
from transformers import AutoTokenizer
# from peft import PeftModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import sacrebleu
import transformers
import vllm

print("--- Environment Check ---")
print(f"Transformers version: {transformers.__version__}")
print(f"vLLM version: {vllm.__version__}")
print("-------------------------")

def extract_answer(response_text):
    """
    Extracts the final answer from the model's response using regex.
    Looks for the LAST occurrence of content between <answer> and </answer> tags.
    """
    # re.findall returns a list of all matches found in the string
    matches = re.findall(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # [-1] grabs the very last match in the list
        return matches[-1].strip()
    
    return ""

def parse_base_model_path(base_model):
    parts = [p for p in base_model.replace("\\", "/").split("/") if p]
    if "merged_models" in parts:
        merged_models_idx = parts.index("merged_models")
        if merged_models_idx + 2 < len(parts):
            return parts[merged_models_idx + 1], parts[-1], True
    return parts[-1] if parts else base_model, None, False

def main():
    parser = argparse.ArgumentParser(description="Test a base or fine-tuned LoRA model on a list of questions.")
    parser.add_argument("--base-model", type=str, required=True, help="The Hugging Face name or local path to the base model.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Path to the directory containing your saved LoRA adapter weights. If not provided, the base model will be used.")
    parser.add_argument("--test-file", type=str, required=True, help="Path to a JSON file containing your test questions.")
    parser.add_argument("--output-file", type=str, default=None, help="Path to save the generated results for later evaluation. If not provided, it will be auto-generated.")
    parser.add_argument("--num-gpus", type=int, default=1, help="tensor parallel size (num GPUs)")
    parser.add_argument("--print-generation", action="store_true", help="Print the full generated response before the extracted final translation.")
    parser.add_argument("--first-3-samples", action="store_true", help="Run inference only on the first 3 items of the test set.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling probability.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument("--min-p", type=float, default=0.0, help="Minimum probability cutoff.")
    parser.add_argument("--max-tokens", type=int, default=15000, help="Maximum number of generated tokens.")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of prompts to send to vLLM at once. Defaults to all prompts.")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Presence penalty for sampling. > 0.0 to add a penalty when a token has appeared once")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty for sampling, > 1.0 to penalizes new tokens based on whether they appear so far.")
    parser.add_argument("--prompt-template", type=str, default="simple_wrapper", choices=["simple_wrapper", "qwen_chat","auto_template"], help="Prompt formatting to use.")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode for qwen_chat prompting.")
    parser.add_argument("--filename-suffix", type=str, default=None, help="Optional suffix appended to the result filename.")


    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer.")

    auto_generated_output_file = args.output_file is None
    base_model_name, merged_model_name, is_merged_model = parse_base_model_path(args.base_model)
    test_set_name = os.path.splitext(os.path.basename(args.test_file))[0]

    # --- Auto-generate output_file name ---
    if args.output_file is None:
            # 3. Combine them, checking if a checkpoint is used
            if args.checkpoint_dir:
                # Normalize the path and replace backslashes to handle cross-platform formatting
                clean_path = os.path.normpath(args.checkpoint_dir).replace("\\", "/")
                # Split the path into parts and filter out empty strings
                parts = [p for p in clean_path.split("/") if p]
                # Take up to the last n parts of the path and join them with an underscore
                ckpt_name = "_".join(parts[-3:]) if parts else "checkpoint"
                ckpt_name = ckpt_name.replace("checkpoint", "ckpt") 
                
                args.output_file = f"{ckpt_name}.json"
            elif is_merged_model:
                args.output_file = f"base_{base_model_name}_{merged_model_name}_{test_set_name}.json"
            else:
                args.output_file = f"base_{base_model_name}_{test_set_name}.json"

    if args.first_3_samples:
        stem, ext = os.path.splitext(args.output_file)
        args.output_file = f"{stem}_3samples{ext}"

    if args.filename_suffix:
        suffix = args.filename_suffix.strip().lstrip("_")
        if suffix:
            stem, ext = os.path.splitext(args.output_file)
            args.output_file = f"{stem}_{suffix}{ext}"

    if auto_generated_output_file:
        print(f"Auto-generated output filename: {args.output_file}\n")

    # --- LOAD MODEL  ---
    print(f"Loading base model: {args.base_model} with vLLM...")
    
    # Check if a checkpoint directory was provided so we can enable LoRA in vLLM
    use_lora = args.checkpoint_dir is not None
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    
    llm = LLM(
        model=args.base_model,
        trust_remote_code=True, # for handling very new models, tell vLLM to download the architecture code directly from the model's repository
        dtype="bfloat16",
        enable_lora=use_lora,
        max_lora_rank=64, # Adjust this if your LoRA rank (r) during training was higher than 64
        tensor_parallel_size=args.num_gpus,
        language_model_only=True, # disable any vision modules for models like Qwen3.5
        max_model_len=32768, # prompt + max_tokens(new generation length)
        enforce_eager=True,
    )

    # Define generation parameters (temperature, max_tokens, etc.)
    # Based on Qwen team's best practices to prevent endless repetitions:
    # Temperature=0.6, TopP=0.95, TopK=20, MinP=0.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        # stop_token_ids=[...] # Optional: add if you want to stop early on specific tokens
    )

    # --- FORMAT PROMPT ---
    # CRITICAL: This must match your training template exactly!
    def format_prompt(instruction):
        if args.prompt_template == "simple_wrapper":
            return f"### Prompt:\n{instruction}\n\n### Answer:\n"

        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )

    # --- LOAD QUESTIONS ---
    print(f"Loading test questions from: {args.test_file}...")
    
    with open(args.test_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)  # Loads the entire JSON file into a Python list 

    if args.first_3_samples:
        dataset = dataset[:3]

    print(f"Successfully loaded {len(dataset)} questions. Starting inference...\n")
    print("="*50)

    # List to store our results
    results = []
    # # Initialize lists for BLEU/chrF calculation ---
    # all_predictions = []
    # all_references = []

    # Prepare batched prompts and run generation ---
    # 1. Format all prompts first
    formatted_prompts = []
    for data in dataset:
        question = data.get('prompt')
        formatted_prompts.append(format_prompt(question))
        
    # 2. Run batched inference via vLLM
    print("Running batched generation...")
    batch_size = args.batch_size or len(formatted_prompts)
    print(f"Using generation batch size: {batch_size}")

    if use_lora:
        print(f"Applying LoRA adapter from: {args.checkpoint_dir}...")
        # vLLM applies the LoRA adapter dynamically during generation
        lora_req = LoRARequest("lora_adapter", 1, args.checkpoint_dir)
    else:
        print("Running inference with the BASE MODEL only.")

    outputs = []
    for start in range(0, len(formatted_prompts), batch_size):
        end = min(start + batch_size, len(formatted_prompts))
        print(f"Generating prompts {start + 1}-{end} of {len(formatted_prompts)}...")
        batch_prompts = formatted_prompts[start:end]
        if use_lora:
            batch_outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_req)
        else:
            batch_outputs = llm.generate(batch_prompts, sampling_params)
        outputs.extend(batch_outputs)
        
    # 3. Process the results
    for i, (data, output) in enumerate(zip(dataset, outputs), 1):
        sent_id = data.get('sent_id')
        question = data.get('prompt')
        gold_translation = data.get('text[eng]',data.get('translation', '')) # both 'text[eng]' and 'text' keys for gold reference
        
        # Extract the generated text from the vLLM output object
        response = output.outputs[0].text
        final_translation = extract_answer(response)
        
        # print out the generation for debugging purposes
        if args.print_generation:
            print(f"Formatted prompt example: {formatted_prompts[0]}")
            print(f"\n[ Question {i}/{len(dataset)} ]")
            # print(f"prompt: {question}")
            print(f"response: {response}")
            print(f"final_translation: {final_translation}")
        
        # Append the current evaluation data to our results list
        results.append({
            "sent_id": sent_id,
            "prompt": question,
            "generated_response": response,
            "final_translation": final_translation,
            "gold_translation": gold_translation,
        })
    # --- MODIFICATION END ---


    # Calculate and print BLEU and chrF scores ---
    # print("\n" + "="*50)
    # print("Calculating sacrebleu metrics...")
    
    # # sacrebleu expects references to be a list of lists (to support multiple references per sentence)
    # bleu = sacrebleu.corpus_bleu(all_predictions, [all_references])
    # chrf = sacrebleu.corpus_chrf(all_predictions, [all_references])
    
    # print(f"Corpus BLEU Score: {bleu.score:.2f}")
    # print(f"Corpus chrF Score: {chrf.score:.2f}")
    # print("="*50 + "\n")

    if not is_merged_model and base_model_name in {"Qwen3-8B", "Qwen3-14B","gemma-4-E2B-it","gemma-4-E4B-it","gemma-4-31B-it"}:
        thinking_suffix = "think" if args.enable_thinking else "no-think"
        base_model_name = f"{base_model_name}-{thinking_suffix}"

    # --- SAVE RESULTS ---
    save_dir = "results" # 1. Define your target directory
    os.makedirs(os.path.join(save_dir, base_model_name, test_set_name), exist_ok=True) # 2. Create the subdirectory if it doesn't already exist
    final_output_path = os.path.join(save_dir, base_model_name, test_set_name, args.output_file) # 3. Safely combine the folder and the filename

    print(f"\nSaving all results to {args.output_file}...")
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print("Inference and saving complete!")

if __name__ == "__main__":
    main()
