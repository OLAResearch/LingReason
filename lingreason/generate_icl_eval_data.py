from .corpus_profiles import (
    get_corpus_profile,
    get_source_text,
    get_translation_text,
    load_dictionary,
)
from .generate_reasoning_traces_from_UD import (
    get_gloss_text,
    get_lemma_text,
    get_reasoning_steps,
    get_subtree_nodes,
)


def component_dictionary_entries(root, language_code, sent):
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

    wordbyword = "\n".join(entries)

    return f"""
Dictionary entries:
{wordbyword}"""


def component_reasoning_scaffold(
    root,
    language_code,
    include_grammar_rules=False,
    max_grammar_rules=3,
    long_rule_once_threshold=400,
    number_placeholders=False,
    mask_final_translation_placeholder=False,
):
    reasoning_scaffold = get_reasoning_steps(
        root,
        language_code,
        include_grammar_rules=include_grammar_rules,
        max_grammar_rules=max_grammar_rules,
        long_rule_once_threshold=long_rule_once_threshold,
        number_placeholders=number_placeholders,
        mask_final_translation_placeholder=mask_final_translation_placeholder,
    )
    return f"""
You are also given a linguistic reasoning guide for this sentence.
Use it as in-context guidance for translating the sentence.
Some placeholders for Lexical Meanings, Phrasal Translations, and the Final Translation are intentionally left unfilled.

Linguistic reasoning guide:
{reasoning_scaffold}"""


def prompt_template_icl_reasoning(src_lang, sent, components_list=None):
    components = "\n".join(components_list or [])
    prompt = f"""
Please help me translate the following sentence from {src_lang} to English:
{sent}

{components}



Your task is to use the dictionary entries and the linguistic reasoning guide above to derive the final English translation.
Proceed step by step through the linguistic reasoning guide and resolve the placeholders in ascending order.

For each lexical-meaning placeholder, such as [Lexical Meaning 1]:
1. Use the provided dictionary entries as the source of possible word meanings.
2. Choose the meaning that best fits the local context.
3. Use the word explanation immediately before the placeholder, including part of speech, lemma, morphology, case, tense, aspect, number, person, polarity, or other grammatical features.
4. If the dictionary gives multiple meanings, prefer the one that is compatible with the morphological and syntactic explanation in the guide.
5. Sometimes there is no dictionary entry for a word. In that case,  try to guess the meaning based on the word form, the context, and the reasoning guide. It could be a proper noun, a loanword, a compound, or a typo.

For each phrasal-translation placeholder, such as [Phrasal Translation 1]:
1. Combine meanings that have already been resolved in earlier placeholders.
2. Use the syntactic relationship described in the guide to decide how the dependent combines with the head.
3. Use word order, case marking, adpositions, auxiliaries, modifiers, subjects, objects, clauses, and any provided grammar notes.
4. Translate the whole subtree named in that line, not just the head word.
5. Make the phrase meaning consistent with the meanings chosen for its component words.

Continue this bottom-up process until you reach [Final Translation].

Important output requirements:
Output first the completed linguistic reasoning guide with all placeholders resolved inside <completed_guide> ... </completed_guide> tags.
Then output only the final English translation inside <answer> ... </answer> tags.
Do not add any text before <completed_guide>, between </completed_guide> and <answer>, or after </answer>.

"""
    return prompt


def generate_icl_eval_prompt(
    language_code,
    sent,
    root,
    include_dict=True,
    include_grammar_rules_in_scaffold=True,
    max_grammar_rules=3,
    long_rule_once_threshold=400,
    number_placeholders=False,
    mask_final_translation_placeholder=False,
    extra_components=None,
):
    components = list(extra_components or [])

    if include_dict:
        if root is None:
            raise ValueError("root is required when include_dict=True")
        components.append(component_dictionary_entries(root, language_code, sent))

    components.append(
        component_reasoning_scaffold(
            root,
            language_code,
            include_grammar_rules=include_grammar_rules_in_scaffold,
            max_grammar_rules=max_grammar_rules,
            long_rule_once_threshold=long_rule_once_threshold,
            number_placeholders=number_placeholders,
            mask_final_translation_placeholder=mask_final_translation_placeholder,
        )
    )

    language_name = get_corpus_profile(language_code).language_name
    return prompt_template_icl_reasoning(language_name, sent, components)


def generate_icl_eval_examples(
    trees,
    language_code,
    include_dict=True,
    include_grammar_rules_in_scaffold=False,
    max_grammar_rules=3,
    long_rule_once_threshold=400,
    number_placeholders=False,
    mask_final_translation_placeholder=False,
):
    examples = []

    for root in trees:
        sent_id = root.metadata["sent_id"]
        sent = get_source_text(root.metadata, language_code)
        gold_translation = get_translation_text(root.metadata, language_code, required=True)

        examples.append({
            "sent_id": sent_id,
            "text": sent,
            "translation": gold_translation,
            "prompt": generate_icl_eval_prompt(
                language_code=language_code,
                sent=sent,
                root=root,
                include_dict=include_dict,
                include_grammar_rules_in_scaffold=include_grammar_rules_in_scaffold,
                max_grammar_rules=max_grammar_rules,
                long_rule_once_threshold=long_rule_once_threshold,
                number_placeholders=number_placeholders,
                mask_final_translation_placeholder=mask_final_translation_placeholder,
            ),
        })

    return examples


def save_icl_eval_examples(examples, output_path):
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)


def print_icl_eval_examples(examples, n=3, start=0):
    for i, ex in enumerate(examples[start:start + n], start=start):
        print("=" * 100)
        print(f"INDEX: {i}")
        print(f"SENT_ID: {ex.get('sent_id')}")
        print(f"TEXT: {ex.get('text')}")
        print(f"GOLD_TRANSLATION: {ex.get('translation')}")
        print("-" * 100)
        print("PROMPT:")
        print(ex.get("prompt", ""))
        print("=" * 100)
        print()
