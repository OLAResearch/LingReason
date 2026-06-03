#!/usr/bin/env python
# coding: utf-8

# Simple benchmarking script that does fine-tuning on a given Hugging
# Face model with IMDB movie reviews
#
# Adapted from the exercises of the LUMI AI workshop course:
# https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop

import argparse
import os
import sys
import time

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
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer,
                          TrainingArguments,
                          EarlyStoppingCallback)
from peft import get_peft_model, LoraConfig
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
import wandb
# import sacrebleu # for testing venv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507", help="The pre-trained model from Hugging Face to use as basis: https://huggingface.co/models")
    parser.add_argument("--output-path", type=str, help="The root directory under which model checkpoints are stored.")
    parser.add_argument("--dataset-path", type=str, default="data_prompt_answer_383.json", help="Path to the JSON dataset file.")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Training batch size, MUST be a multiple of 8 if using 8 GPUs")
    parser.add_argument("--num-workers", type=int, default=7, help="The number of CPU worker processes to use.")
    parser.add_argument("--max-steps", type=int, default=400, help="The number of training steps.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Optional Trainer checkpoint path to resume from.")
    parser.add_argument("--prompt-template", type=str, default="simple_wrapper", choices=["simple_wrapper", "qwen_chat"], help="Prompt formatting to use.")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode for qwen_chat prompting.")
    parser.add_argument("--peft", action='store_true', help="Use PEFT: https://huggingface.co/docs/peft/index")
    parser.add_argument("--4bit", dest="bnb_4bit", action='store_true', help="Use 4bit quantization with bitsandbytes: https://huggingface.co/docs/bitsandbytes/main/en/index")
    args, _ = parser.parse_known_args()

    # Read the environment variables provided by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    # Then we determine the device on which to train the model.
    if rank == 0:
        print("Using PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        print(f"Using GPU {local_rank}, device name: {torch.cuda.get_device_name(device)}")
    else:
        print(f"No GPU found, using CPU instead. (Rank: {local_rank})")
        device = torch.device("cpu")

    if rank == 0 and args.batch_size % world_size != 0:
        print(f"ERROR: batch_size={args.batch_size} has to be a multiple of "
              f"the number of GPUs={world_size}!")
        sys.exit(1)

    # 1. Extract just the model name (e.g., "Qwen/Qwen3-4B" -> "Qwen3-4B")
    model_name = args.model.split("/")[-1]
    if model_name in {"Qwen3-8B", "Qwen3-14B"}:
        thinking_suffix = "think" if args.enable_thinking else "no-think"
        model_name = f"{model_name}-{thinking_suffix}"
    dataset_name = os.path.basename(args.dataset_path).split('.')[0]

    # this is where trained model and checkpoints will go
    output_dir = os.path.join(args.output_path, model_name, f"SFT@{dataset_name}")

    # Load the data set
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    # Use 100% of the dataset for training, matching the RFT setup.
    train_dataset = dataset
    # Split it into train and validation
    # splits = dataset.train_test_split(test_size=0.1, seed=42)
    # train_dataset = splits["train"]
    # eval_dataset = splits["test"]

    # #### Loading the model
    # Let's start with getting the appropriate tokenizer.
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = tokenizer.special_tokens_map

    # Load the actual base model from Hugging Face
    if rank == 0:
        print("Loading model and tokenizer")

    quantization_config = None
    if args.bnb_4bit: # quantization might not work well on LUMI with AMD GPUs?
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        quantization_config = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        # attn_implementation="flash_attention_2",
        )
    
    model.enable_input_require_grads()# force Gradient Checkpointing even when PEFT (LoRA) PEFT freezes 99% of the weights
    
    model.config.use_cache = False # Silence the use_cache warning

    if args.peft:
        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
        #     lora_dropout=0.1
        # )

        # LoRA config from here:
        # https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_fsdp_qlora.py#L128
        peft_config = LoraConfig(
            lora_alpha=8, # how strongly the adapter is applied
            lora_dropout=0.05,
            r=16, # how much the adapter can learn
            bias="none",
            target_modules="all-linear",
            exclude_modules=r".*visual\..*",  # exclude visual modules for models like Qwen3.5
            task_type="CAUSAL_LM",
            # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
            )
        model = get_peft_model(model, peft_config)
        print("Using PEFT")
        model.print_trainable_parameters()

    #model.to(device)
    stop = time.time()
    if rank == 0:
        print(f"Loading model and tokenizer took: {stop-start:.2f} seconds")

    train_batch_size = args.batch_size
    # eval_batch_size = args.batch_size

    training_args = TrainingArguments(
        output_dir=output_dir,
        # overwrite_output_dir=not args.resume,
        # save_strategy="no",  # good for testing
        save_strategy="steps",   # use these if you actually want to save the model
        save_steps=200,
        eval_strategy="no",
        # eval_strategy="steps",
        # eval_steps=200,  # compute validation loss every ... steps # <-- MUST MATCH eval_steps for EarlyStoppingCallback to work properly
        save_total_limit=16,

        # --- EARLY STOPPING TRIGGERS ---
        # load_best_model_at_end=True,       # Required for early stopping
        # metric_for_best_model="eval_loss", # Tells it what metric to watch
        # greater_is_better=False,           # False means we want the loss to go DOWN

        logging_steps=50,          # Print training loss to the console every ... steps
        learning_rate=1e-5,
        weight_decay=0.01,
        bf16=True,  # use 16-bit floating point precision
        # divide the total training batch size by the number of GCDs for the per-device batch size
        per_device_train_batch_size=train_batch_size // world_size,
        # per_device_eval_batch_size=eval_batch_size,
        # per_device_eval_batch_size=1, # To avoid OOM during Eval: Evaluation does not need to process large batches simultaneously like training does. We can just tell the Trainer to evaluate the validation set one example at a time, (do not use eval_batch_size)
        # eval_accumulation_steps=1,    # Moves eval results to RAM to prevent GPU buildup

        max_steps=args.max_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        # report_to=["tensorboard"],  # log statistics for tensorboard
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        report_to="wandb",
        run_name=f"{model_name}#{dataset_name}",
    )

    # #### Setting up preprocessing of training data
    max_length = 2**13 # cap total prompt+output to 16k (2**14=16384) tokens  # as cmd argument

    def format_prompt(prompt_text):
        if args.prompt_template == "simple_wrapper":
            return f"### Prompt:\n{prompt_text}\n\n### Answer:\n"

        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )

    def format_and_tokenize(example):
        prompt_text = format_prompt(example['prompt'])
        formatted_text = f"{prompt_text}{example['answer']}{tokenizer.eos_token}"

        tokenized = tokenizer(
            formatted_text,
            max_length=max_length,
            truncation=True,
            padding=False # Padding is usually handled by the collator later
        )
        prompt_tokenized = tokenizer(
            prompt_text,
            max_length=max_length,
            truncation=True,
            padding=False
        )
        prompt_length = min(len(prompt_tokenized["input_ids"]), len(tokenized["input_ids"]))
        tokenized["completion_mask"] = [0] * prompt_length + [1] * (len(tokenized["input_ids"]) - prompt_length)

        return tokenized

    # Apply this to your train dataset
    train_dataset_tok = train_dataset.map(
        format_and_tokenize,
        remove_columns=train_dataset.column_names,# remove original column names in the json file such as 'prompt' and 'answer', we only need the tokenized input_ids and attention_mask for training
        batched=False, # <-- MUST BE FALSE for the specific formatting function, format_and_tokenize function is written to handle one example at a time
        batch_size=training_args.train_batch_size,
        num_proc=training_args.dataloader_num_workers,
    )

    # eval_dataset_tok = eval_dataset.map(
    #     format_and_tokenize,
    #     remove_columns=eval_dataset.column_names,
    #     batched=False, # <-- MUST BE FALSE for the specific formatting function, format_and_tokenize function is written to handle one example at a time
    #     num_proc=training_args.dataloader_num_workers,
    # )

    collator = DataCollatorForLanguageModeling( #only let the model to learn how to generate the response/completion, not the prompt
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset_tok,
        # eval_dataset=eval_dataset_tok,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]#If the eval loss goes up for _ consecutive evaluations, stop the training
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # > Add Weights & Biases for visualization

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(output_dir)
    if rank == 0:
        print()
        print("Training done, you can find the final model (and checkpoints) in", output_dir)
