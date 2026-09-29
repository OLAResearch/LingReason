from pathlib import Path

from .corpus_profiles import (
    CORPUS_PROFILES,
    get_corpus_profile,
    get_source_text as get_profile_source_text,
    get_translation_text as get_profile_translation_text,
    load_dictionary,
)
from .generate_reasoning_traces_from_UD import (
    get_gloss_text,
    get_lemma_text,
    get_reasoning_steps_with_rule_ids,
    get_subtree_nodes,
    load_grammar_rules,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


SRC_NAMES = {
    code: profile.language_name
    for code, profile in CORPUS_PROFILES.items()
}

GRAMMAR_PATHS = {
    code: profile.grammar_sketch_path
    for code, profile in CORPUS_PROFILES.items()
    if profile.grammar_sketch_path is not None
}

DICT_PATHS = {
    code: profile.dictionary_path
    for code, profile in CORPUS_PROFILES.items()
    if profile.dictionary_path is not None
}

def prompt_template(src_lang, sent, components_list=None):
    components = '\n'.join(components_list or [])
    prompt = f"""
Please help me translate the following sentence from {src_lang} to English:
{sent}

{components}

Using all the information provided above, you should proceed step-by-step: first determine the meaning and part-of-speech of each word; then identify the syntactic relationships among the words; then based on the syntactic relationships, combine the meanings of individual words to get phrase meanings; and continue this process until you eventually derive the meaning of the whole sentence. 

Your response must contain exactly two sections:
1. Step-by-step reasoning inside <think> ... </think>
2. The final English translation inside <answer> ... </answer>
Do not add any extra text outside these tags.

Remember your source sentence is: {sent}
"""
    return prompt


def prompt_template_no_thinking(src_lang, sent, components_list=None):
    components = '\n'.join(components_list or [])
    prompt = f"""
Please help me translate the following sentence from {src_lang} to English:
{sent}

{components}

Using all the information provided above, your task is to translate the source sentence into English.

Please put your final English translation inside <answer> ... </answer>
Do not add any extra text outside these tags.

Remember your source sentence is: {sent}
"""
    return prompt


def get_lemmas(root,language_code):
    lemmas = []
    for node in get_subtree_nodes(root):
        lemma = get_lemma_text(node.token, language_code)
        if lemma != "_" and lemma not in lemmas:
            lemmas.append(lemma)
    return lemmas


def component_dict_entries(language_code, root):
    lemma_dict = load_dictionary(language_code)
    entries = []
    seen_lemmas = set()
    for node in get_subtree_nodes(root):
        lemma = get_lemma_text(node.token, language_code)
        if lemma == "_" or lemma in seen_lemmas:
            continue
        seen_lemmas.add(lemma)
        gloss = get_gloss_text(node.token, language_code, lemma_dict).strip()
        entries.append(f"{lemma}: {gloss or 'not found in dictionary'}")
    wordbyword = '\n'.join(entries)
    src_lang = get_corpus_profile(language_code).language_name
    return f"""
For the translation task, you are given the dictionary entries for each individual word of the {src_lang} sentence.
Some words may be polysemous and there might be multiple possible English translations. In such case, please choose the most appropriate one.

Here are the dictionary entries for each individual word in the source sentence:
{wordbyword}"""




def component_grammar(language_code):
    profile = get_corpus_profile(language_code)
    if profile.grammar_sketch_path is None:
        raise ValueError(f"No grammar sketch configured for {profile.corpus_id}")
    with profile.grammar_sketch_path.open("r", encoding="utf-8") as f:
        grammar_sketch = f.read()
    return f"""
You are also given this grammatical sketch below. Feel free to rely on the this grammatical sketch in your translation task:
{grammar_sketch}"""


def component_grammar_from_rule_ids(language_code, rule_ids):
    rule_ids = set(rule_ids or [])
    rules = load_grammar_rules(language_code)
    grammar_rules = "\n".join(
        f"- {rule['rule_text']}"
        for rule in rules
        if rule["id"] in rule_ids
    )
    return f"""
You are also given the grammar rules that are directly relevant to this sentence:
{grammar_rules}"""


def generate_translation_prompt(
    language_code,
    sent,
    root=None,
    include_dict=False,
    grammar_mode="sketch",
    grammar_rule_ids=None,
    thinking=True,
    extra_components=None,
):
    components = list(extra_components or [])

    if include_dict:
        if root is None:
            raise ValueError("root is required when include_dict=True")
        components.append(component_dict_entries(language_code, root))

    if grammar_mode is not None:
        if grammar_mode == "used_rules":
            components.append(component_grammar_from_rule_ids(language_code, grammar_rule_ids))
        elif grammar_mode == "sketch":
            components.append(component_grammar(language_code))
        else:
            raise ValueError(f"Unknown grammar_mode: {grammar_mode}")

    template = prompt_template if thinking else prompt_template_no_thinking
    return template(get_corpus_profile(language_code).language_name, sent, components)


def format_sft_answer(translation, reasoning_trace=None, thinking=True):
    translation = translation.strip()
    if not thinking:
        return f"<answer>\n{translation}\n</answer>"

    if not reasoning_trace:
        raise ValueError("reasoning_trace is required when thinking=True")

    return (
        f"<think>\n{reasoning_trace.strip()}\n</think>\n"
        f"<answer>\n{translation}\n</answer>"
    )


def load_reasoning_results(results_path, valid_only=True):
    import json

    with open(results_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    results = {}
    for item in items:
        if valid_only and item.get("valid_format") is not True:
            continue
        if item.get("sent_id") and item.get("generated_text"):
            results[item["sent_id"]] = item
    return results


def get_source_text(root, language_code):
    return get_profile_source_text(root.metadata, language_code)


def get_gold_translation(root, language_code):
    return get_profile_translation_text(root.metadata, language_code, required=True)


def generate_sft_examples(
    trees,
    language_code,
    reasoning_results_path=None,
    include_dict=False,
    grammar_mode="used_rules",
    max_grammar_rules=3,
    long_rule_once_threshold=400,
    thinking=True,
    valid_only=True,
):
    if thinking and not reasoning_results_path:
        raise ValueError("reasoning_results_path is required when thinking=True")

    reasoning_results = (
        load_reasoning_results(reasoning_results_path, valid_only=valid_only)
        if reasoning_results_path
        else {}
    )

    examples = []
    for root in trees:
        sent_id = root.metadata["sent_id"]
        if thinking and sent_id not in reasoning_results:
            continue
        if reasoning_results_path and sent_id not in reasoning_results:
            continue

        sent = get_source_text(root, language_code)
        translation = get_gold_translation(root, language_code)
        reasoning_trace = reasoning_results.get(sent_id, {}).get("generated_text")
        grammar_rule_ids = set()
        if grammar_mode == "used_rules":
            _, grammar_rule_ids = get_reasoning_steps_with_rule_ids(
                root,
                language_code,
                include_grammar_rules=True,
                max_grammar_rules=max_grammar_rules,
                long_rule_once_threshold=long_rule_once_threshold,
            )

        examples.append({
            "sent_id": sent_id,
            "text": sent,
            "translation": translation,
            "prompt": generate_translation_prompt(
                language_code=language_code,
                sent=sent,
                root=root,
                include_dict=include_dict,
                grammar_mode=grammar_mode,
                grammar_rule_ids=grammar_rule_ids,
                thinking=thinking,
            ),
            "answer": format_sft_answer(
                translation=translation,
                reasoning_trace=reasoning_trace,
                thinking=thinking,
            ),
        })
    return examples

def split_sft_examples(examples, train_ratio=0.8, seed=42):
    import random

    examples = list(examples)
    random.seed(seed)
    random.shuffle(examples)

    split_index = int(len(examples) * train_ratio)
    return examples[:split_index], examples[split_index:]


def save_sft_examples(examples, output_path):
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)

def print_sft_examples(examples, n=3, start=0):
    for i, ex in enumerate(examples[start:start + n], start=start):
        print("=" * 100)
        print(f"INDEX: {i}")
        print(f"SENT_ID: {ex.get('sent_id')}")
        print(f"TEXT: {ex.get('text')}")
        print(f"TRANSLATION: {ex.get('translation')}")
        print("-" * 100)
        print("PROMPT:")
        print(ex.get("prompt", ""))
        print("-" * 100)
        print("ANSWER:")
        print(ex.get("answer", ""))
        print("=" * 100)
        print()
