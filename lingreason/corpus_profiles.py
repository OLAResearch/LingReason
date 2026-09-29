"""Corpus-specific configuration shared by data-generation pipelines."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CorpusProfile:
    corpus_id: str
    language_code: str
    language_name: str
    source_metadata_keys: Tuple[str, ...]
    translation_metadata_keys: Tuple[str, ...]
    word_misc_key: Optional[str] = None
    lemma_misc_key: Optional[str] = None
    gloss_misc_key: Optional[str] = "Gloss"
    gloss_sources: Tuple[str, ...] = ("misc", "dictionary")
    dictionary_path: Optional[Path] = None
    grammar_rules_path: Optional[Path] = None
    grammar_sketch_path: Optional[Path] = None
    adapter: str = "generic"


CORPUS_PROFILES: Dict[str, CorpusProfile] = {
    "ctn": CorpusProfile(
        corpus_id="ctn_ctntb",
        language_code="ctn",
        language_name="Chintang",
        source_metadata_keys=("text",),
        translation_metadata_keys=("english",),
        dictionary_path=PROJECT_ROOT / "dicts_generated_from_UD" / "dict_UD_gloss_ctn.json",
        grammar_rules_path=PROJECT_ROOT / "gram_rules" / "gram_rules_ctn.json",
        grammar_sketch_path=PROJECT_ROOT / "gram_sketches" / "gram_sketch_ctn.txt",
    ),
    "xcl": CorpusProfile(
        corpus_id="xcl_caval",
        language_code="xcl",
        language_name="Classical Armenian",
        source_metadata_keys=("transliterated_text", "text"),
        translation_metadata_keys=("translated_text",),
        word_misc_key="Translit",
        lemma_misc_key="LTranslit",
        dictionary_path=PROJECT_ROOT / "dicts_generated_from_UD" / "dict_UD_gloss_xcl.json",
        grammar_rules_path=PROJECT_ROOT / "gram_rules" / "gram_rules_xcl.json",
        grammar_sketch_path=PROJECT_ROOT / "gram_sketches" / "gram_sketch_xcl.txt",
    ),
}


def register_corpus_profile(profile: CorpusProfile, *aliases: str) -> None:
    """Register a profile, normally from a private or project-local extension."""
    for key in (profile.language_code, profile.corpus_id, *aliases):
        CORPUS_PROFILES[key] = profile


def get_corpus_profile(identifier: str) -> CorpusProfile:
    try:
        return CORPUS_PROFILES[identifier]
    except KeyError:
        for profile in CORPUS_PROFILES.values():
            if profile.corpus_id == identifier:
                return profile
        supported = ", ".join(sorted(CORPUS_PROFILES))
        raise ValueError(
            f"Unknown corpus or language code {identifier!r}. Registered profiles: {supported}"
        )


def get_metadata_value(metadata, keys: Tuple[str, ...], field_name: str, required=True):
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    if required:
        raise ValueError(
            f"Missing {field_name}; expected one of these metadata fields: {', '.join(keys)}"
        )
    return " "


def get_source_text(metadata, identifier: str) -> str:
    profile = get_corpus_profile(identifier)
    return get_metadata_value(metadata, profile.source_metadata_keys, "source text")


def get_translation_text(metadata, identifier: str, required=False) -> str:
    profile = get_corpus_profile(identifier)
    return get_metadata_value(
        metadata,
        profile.translation_metadata_keys,
        "English translation",
        required=required,
    )


_DICTIONARY_CACHE = {}


def load_dictionary(identifier: str, required=False):
    profile = get_corpus_profile(identifier)
    path = profile.dictionary_path
    if path is None or not path.exists():
        if required:
            raise FileNotFoundError(
                f"No dictionary is available for {profile.corpus_id}. "
                f"Configure dictionary_path in its CorpusProfile."
            )
        return {}
    if path not in _DICTIONARY_CACHE:
        with path.open("r", encoding="utf-8") as f:
            _DICTIONARY_CACHE[path] = json.load(f)
    return _DICTIONARY_CACHE[path]
