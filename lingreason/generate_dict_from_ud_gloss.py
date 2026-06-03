import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

def clean_string(text, lemma=None):
    if lemma and lemma[:1].isupper() and text[:1].isupper():
        return re.split(r"[.-]", text, maxsplit=1)[0].replace("_", " ")

    # 1. If there are NO lowercase letters, keep as-is.(This is preserve some abbreviations for grammatical elements.)
    # 1. The "Pure Tag" Check
    # Temporarily remove all tags (anything starting with a Capital/Number + its lowercase trail)
    text_without_tags = re.sub(r'[A-Z0-9][a-z]*', '', text)
    
    # If the remaining string has no lowercase letters, the original string
    # was comprised completely of tags and symbols (like "3Sing.Obj" or "VNOUN").
    if not re.search(r'[a-z]', text_without_tags):
        return text

    # 2. Target and remove mixed-case tags and numbers (e.g., 'Vnoun', 'Sing', 'Gen', '2')
    # [A-Z0-9] looks for the start of a tag (a capital letter or number)
    # [a-z]* grabs any lowercase letters that belong to that specific tag
    text = re.sub(r'[A-Z0-9][a-z]*', '', text)


    # 3. Use regex to find hyphens that are NOT between lowercase letters
    # We use a lambda to protect the "good" hyphens
    # Pattern explanation:
    # Keep matches of lowercase-hyphen-lowercase,
    # but replace any OTHER hyphen with an empty string.

    # First, let's remove symbols/numbers/caps but KEEP underscores, spaces, and hyphens
    text = re.sub(r'[^a-z_ \-]', '', text)

    # Now, handle the hyphen specific logic:
    # This regex matches a hyphen only if it is NOT preceded by a-z AND followed by a-z
    # We use a negative lookbehind (?<![a-z]) and negative lookahead (?![a-z])
    # However, it's cleaner to use a substitution that targets hyphens
    # and only keeps them if the surrounding context matches.

    def hyphen_filter(match):
        s = match.group(0)
        # If hyphen is between two lowercase letters, keep it
        return re.sub(r'(?<![a-z])-|-(?![a-z])', '', s)

    # We refine the string: remove hyphens that don't have letters on both sides
    text = re.sub(r'(?<![a-z])-|-(?![a-z])', '', text)

    # Handle the underscore/hyphen co-occurrence rule
    # If both are present, we don't replace _ with space.
    # Otherwise, we do.
    if not ('-' in text and '_' in text):
        text = text.replace('_', ' ')

    return text


def load_ud_sentences(folder_path):
    try:
        from conllu import parse
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'conllu'. Run this script with the project .venv "
            "or install it with: pip install conllu"
        ) from exc

    folder = Path(folder_path)
    conllu_files = sorted(folder.glob("*.conllu"))
    if not conllu_files:
        raise FileNotFoundError(f"No .conllu files found in {folder}")

    all_data = "\n\n".join(
        conllu_file.read_text(encoding="utf-8")
        for conllu_file in conllu_files
    )
    return parse(all_data), conllu_files

def load_ud_trees(folder_path):
    try:
        from conllu import parse_tree
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'conllu'. Run this script with the project .venv "
            "or install it with: pip install conllu"
        ) from exc

    folder = Path(folder_path)
    conllu_files = sorted(folder.glob("*.conllu"))
    if not conllu_files:
        raise FileNotFoundError(f"No .conllu files found in {folder}")

    all_data = "\n\n".join(
        conllu_file.read_text(encoding="utf-8")
        for conllu_file in conllu_files
    )
    return parse_tree(all_data), conllu_files

def should_use_ltranslit(folder_path, conllu_files):
    folder_name = Path(folder_path).name
    return (
        folder_name == "UD_Classical_Armenian-CAVaL"
        or any(conllu_file.name.startswith("xcl_") for conllu_file in conllu_files)
    )


def get_key(token, use_ltranslit=False):
    misc = token.get("misc") or {}

    if use_ltranslit:
        return misc.get("LTranslit") or token.get("lemma")

    return token.get("lemma")


def has_usable_gloss(token):
    misc = token.get("misc")
    if not misc or misc == "_":
        return False

    return (
        token.get("lemma") != "_"
        and token.get("upos") != "PUNCT"
        and "Gloss" in misc
        and misc["Gloss"] is not None
    )


def build_gloss_dict(sentences, use_ltranslit=False):
    lemma_gloss_dict = defaultdict(set)

    for sentence in sentences:
        for token in sentence:
            if not has_usable_gloss(token):
                continue

            lemma = get_key(token, use_ltranslit=use_ltranslit)
            if not lemma or lemma == "_":
                continue

            cleaned_gloss = clean_string(token["misc"]["Gloss"], lemma)
            lemma_gloss_dict[lemma].add(cleaned_gloss)

    return {
        lemma: ", ".join(sorted(gloss_set))
        for lemma, gloss_set in sorted(lemma_gloss_dict.items())
    }


def get_language_code(conllu_files):
    first_file = conllu_files[0].name
    match = re.match(r"([A-Za-z]{3})", first_file)
    if not match:
        raise ValueError(f"Could not infer three-letter language code from {first_file}")
    return match.group(1).lower()


def default_output_path(conllu_files):
    language_code = get_language_code(conllu_files)
    return Path(f"dict_UD_gloss_{language_code}.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a lemma-to-gloss JSON dictionary from UD .conllu files."
    )
    parser.add_argument(
        "folder",
        help="Folder containing one or more .conllu files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path. Defaults to final_gloss_dict_<language_code>.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sentences, conllu_files = load_ud_sentences(args.folder)
    use_ltranslit = should_use_ltranslit(args.folder, conllu_files)
    gloss_dict = build_gloss_dict(sentences, use_ltranslit=use_ltranslit)
    output_path = args.output or default_output_path(conllu_files)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(gloss_dict, f, ensure_ascii=False, indent=4)

    print(f"Read {len(conllu_files)} .conllu files")
    print(f"Using {'LTranslit' if use_ltranslit else 'lemma'} as dictionary keys")
    print(f"Wrote {len(gloss_dict)} entries to {output_path}")


if __name__ == "__main__":
    main()
