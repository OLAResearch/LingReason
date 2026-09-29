import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .corpus_profiles import CORPUS_PROFILES, get_corpus_profile

try:
    from google import genai
    from google.genai import types
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'google-genai'. Install it with: pip install google-genai"
    ) from exc


def load_prompt_dicts(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_results(path, results):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def default_output_path(input_json, model):
    safe_model = model.replace("/", "_").replace(":", "_")
    return Path(input_json).with_name(f"{Path(input_json).stem}_{safe_model}_results.json")


def infer_language_code(input_json):
    stem = Path(input_json).stem
    language_codes = sorted(
        {profile.language_code for profile in CORPUS_PROFILES.values()},
        key=len,
        reverse=True,
    )
    for language_code in language_codes:
        if re.search(rf"(^|_){language_code}($|_)", stem):
            return language_code
    return None


def valid_format(text: str) -> bool:
    if not re.search(r"Step\s+\d+:", text):
        return False
    if "[Lexical Meaning" in text or "[Phrasal Translation" in text:
        return False
    if "Putting all these pieces together, the whole sentence" not in text:
        return False
    if not re.search(r"\[[^\]]+\]", text):
        return False
    if re.search(r"([ .,!?;:()\[\]\-])\1{4,}", text):
        return False
    if re.search(r"(.{1,10}?)(?:\s+\1){4,}", text):
        return False
    return True


def get_reasoning_steps_from_prompt(prompt):
    marker = "Reasoning steps with placeholders:"
    if marker not in prompt:
        return None
    return prompt.split(marker, 1)[1].strip()


def is_quota_error(exc) -> bool:
    text = repr(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


def generate_text(client, model, prompt, max_output_tokens, temperature, top_p, thinking_level):
    config_kwargs = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if thinking_level != "none":
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    config = types.GenerateContentConfig(**config_kwargs)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text or ""


def generate_text_with_api_retries(
    client,
    model,
    prompt,
    max_output_tokens,
    temperature,
    top_p,
    thinking_level,
    max_retries,
):
    for attempt in range(max_retries + 1):
        try:
            return generate_text(
                client=client,
                model=model,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                thinking_level=thinking_level,
            ).strip()
        except Exception as exc:
            if is_quota_error(exc):
                raise
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)


def should_skip(item, overwrite):
    return item.get("generated_text") and item.get("valid_format") is not False and not overwrite


def get_last_valid_index(items):
    last_valid_index = 0
    for idx, item in enumerate(items, 1):
        if item.get("valid_format") is True:
            last_valid_index = idx
    return last_valid_index


def process_item(idx, item, args):
    prompt = item["prompt"]
    reasoning_steps = get_reasoning_steps_from_prompt(prompt)
    is_generic = get_corpus_profile(args.language_code).adapter == "generic"
    if is_generic and reasoning_steps and "[Phrasal Translation]" not in reasoning_steps:
        item["generated_text"] = reasoning_steps
        item["valid_format"] = valid_format(reasoning_steps)
        item["skipped_generation"] = True
        item.pop("generation_error", None)
        item.pop("stop_reason", None)
        return idx, item

    client = genai.Client()
    last_error = None
    generated_text = None

    for format_attempt in range(args.max_format_retries + 1):
        try:
            generated_text = generate_text_with_api_retries(
                client=client,
                model=args.model,
                prompt=prompt,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                thinking_level=args.thinking_level,
                max_retries=args.max_retries,
            )
        except Exception as exc:
            last_error = exc
            generated_text = None
            break

        if valid_format(generated_text):
            break

        tqdm.write(
            f"Invalid format for sent_id={item.get('sent_id')} "
            f"(attempt {format_attempt + 1}/{args.max_format_retries + 1})"
        )

    item["generated_text"] = generated_text
    if generated_text is None:
        item["generation_error"] = repr(last_error)
        item["stop_reason"] = "quota_exhausted" if last_error and is_quota_error(last_error) else "generation_error"
        tqdm.write(f"Failed sent_id={item.get('sent_id')} after retries: {last_error}")
    else:
        item.pop("generation_error", None)
        item.pop("stop_reason", None)
        item.pop("skipped_generation", None)
        item["valid_format"] = valid_format(generated_text)

    if args.sleep_seconds:
        time.sleep(args.sleep_seconds)

    return idx, item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="JSON file produced by save_prompts_for_llm_fill_in_place_holders().",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Where to save results. Defaults to <input>_<model>_results.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-flash-lite-preview",
        help="Gemini model name.",
    )
    parser.add_argument("--max_output_tokens", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--thinking_level",
        type=str,
        default="low",
        choices=["none", "low", "high"],
        help="Gemini 3 thinking level. Use 'high' for best quality.",
    )
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_format_retries", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--resume-from-last-valid",
        action="store_true",
        help="Start from the item after the last item with valid_format=true in the output JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate items that already have generated_text.",
    )
    args = parser.parse_args()
    args.language_code = infer_language_code(args.input_json)

    output_json = args.output_json or default_output_path(args.input_json, args.model)
    prompt_dicts = load_prompt_dicts(args.input_json)
    if Path(output_json).exists() and not args.overwrite:
        prompt_dicts = load_prompt_dicts(output_json)

    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    start_index = 1
    if args.resume_from_last_valid:
        last_valid_index = get_last_valid_index(prompt_dicts)
        start_index = last_valid_index + 1
        print(f"Resuming from last valid item: last_valid_index={last_valid_index}, start_index={start_index}")

    pending_items = [
        (idx, item)
        for idx, item in enumerate(prompt_dicts, 1)
        if idx >= start_index and not should_skip(item, args.overwrite)
    ]

    print(f"Loaded {len(prompt_dicts)} prompts.")
    print(f"Inferred language_code={args.language_code}. Generic cases without [Phrasal Translation] will be returned directly.")
    print(f"Processing {len(pending_items)} prompts with concurrency={args.concurrency}.")
    print(f"Saving results to {output_json}")

    stop_generation = False
    with tqdm(total=len(pending_items), desc="Generating") as progress:
        for batch_start in range(0, len(pending_items), args.concurrency):
            if stop_generation:
                break

            batch = pending_items[batch_start: batch_start + args.concurrency]
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [
                    executor.submit(process_item, idx, item, args)
                    for idx, item in batch
                ]
                for future in as_completed(futures):
                    idx, item = future.result()
                    save_results(output_json, prompt_dicts)
                    tqdm.write(f"{idx}/{len(prompt_dicts)} sent_id={item.get('sent_id')}")
                    progress.update(1)

                    if item.get("stop_reason") == "quota_exhausted":
                        tqdm.write("Quota exhausted. Stopping generation.")
                        stop_generation = True

            if stop_generation:
                break

    print("All results saved.")


if __name__ == "__main__":
    main()
