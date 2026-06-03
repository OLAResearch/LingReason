# For the documentation about collu node, see: https://github.com/EmilStenstrom/conllu/blob/master/README.md
from pathlib import Path

from .generate_dict_from_ud_gloss import clean_string

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def number_reasoning_placeholders(text):
    import re

    counts = {
        "Lexical Meaning": 0,
        "Phrasal Translation": 0,
    }

    def replace(match):
        label = match.group(1)
        counts[label] += 1
        return f"[{label} {counts[label]}]"

    return re.sub(r"\[(Lexical Meaning|Phrasal Translation)\]", replace, text)


def mask_final_translation(text):
    import re

    return re.sub(
        r"(translates to: )'.*'(\n|$)",
        r"\1[Final Translation]\2",
        text,
    )


def apply_reasoning_output_options(
    output,
    number_placeholders=False,
    mask_final_translation_placeholder=False,
):
    if mask_final_translation_placeholder:
        output = mask_final_translation(output)
    if number_placeholders:
        output = number_reasoning_placeholders(output)
    return output


def transliterate_classical_armenian(text: str) -> str:
    mapping = {
        "Ա": "A", "ա": "a", "Բ": "B", "բ": "b", "Գ": "G", "գ": "g",
        "Դ": "D", "դ": "d", "Ե": "E", "ե": "e", "Զ": "Z", "զ": "z",
        "Է": "Ē", "է": "ē", "Ը": "Ə", "ը": "ə",
        "Թ": "Tʻ", "թ": "tʻ", "Ժ": "Ž", "ժ": "ž",
        "Ի": "I", "ի": "i", "Լ": "L", "լ": "l", "Խ": "X", "խ": "x",
        "Ծ": "C", "ծ": "c", "Կ": "K", "կ": "k", "Հ": "H", "հ": "h",
        "Ձ": "J", "ձ": "j", "Ղ": "Ł", "ղ": "ł", "Ճ": "Č", "ճ": "č",
        "Մ": "M", "մ": "m", "Յ": "Y", "յ": "y", "Ն": "N", "ն": "n",
        "Շ": "Š", "շ": "š", "Ո": "O", "ո": "o",
        "Չ": "Čʻ", "չ": "čʻ", "Պ": "P", "պ": "p",
        "Ջ": "J̌", "ջ": "ǰ", "Ռ": "Ṙ", "ռ": "ṙ",
        "Ս": "S", "ս": "s", "Վ": "V", "վ": "v", "Տ": "T", "տ": "t",
        "Ր": "R", "ր": "r", "Ց": "Cʻ", "ց": "cʻ",
        "Ւ": "W", "ւ": "w", "Փ": "Pʻ", "փ": "pʻ",
        "Ք": "Kʻ", "ք": "kʻ", "Ֆ": "F", "ֆ": "f",
        "Օ": "O", "օ": "o", "և": "ew",
        "՝": ";", "՞": "?", "՛": "!", "։": ".",
        ":": ".", ".": ":",
    }

    out = []
    i = 0
    while i < len(text):
        pair = text[i:i + 2]
        if pair in {"ու", "ոՒ"}:
            out.append("ow")
            i += 2
        elif pair == "Ու":
            out.append("Ow")
            i += 2
        elif pair == "ՈՒ":
            out.append("OW")
            i += 2
        else:
            out.append(mapping.get(text[i], text[i]))
            i += 1

    return "".join(out)




COMMON_ABBREV_MAP = {
    # A.1 Universal POS Tags (17)
    "ADJ": "adjective",
    "ADP": "postposition", #"adposition"
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",

    # A.2.1 Universal Dependency Relations (30)
    "acl": "clausal modifier of noun",
    "advcl": "adverbial clause modifier",
    "amod": "adjectival modifier",
    "advmod": "adverbial modifier",
    "appos": "appositive",
    "aux": "auxiliary",
    "case": "case marking",
    "cc": "coordinating conjunction",
    "ccomp": "clausal complement",
    "clf": "classifier",
    "compound": "compound",
    "conj": "conjunct",
    "cop": "copula",
    "csubj": "clausal subject",
    "det": "determiner",
    "discourse": "discourse element",
    "fixed": "fixed multiword expression",
    "flat": "flat multiword expression",
    "iobj": "indirect object",
    "mark": "marker",
    "nmod": "nominal modifier",
    "nsubj": "nominal subject",
    "nummod": "numeric modifier",
    "obj": "object",
    "obl": "oblique nominal",
    "parataxis": "parataxis",
    "punct": "punctuation",
    "root": "root",
    "vocative": "vocative",
    "xcomp": "open clausal complement",

    # A.2.2 Relation Subtypes
    "acl:relcl": "relative clause modifier",
    "flat:name": "flat name",
    "flat:num": "flat multiword number",
    "mark:adv": "adverbial marker",
    "mark:plur": "plural marker",
    "mark:rel": "relative marker",
    "nmod:poss": "possessive nominal modifier",
    "nmod:range": "range nominal modifier",
    "nsubj:pass": "passive nominal subject",
    "obl:loc": "locative oblique",
    "obl:tmod": "temporal modifier",
    "obl:lmod": "locative modifier",
    "obj:pass": "passive object",
    # A.3 Features and Values
    "Abbr": "abbreviation",
    "Aspect": "aspect",
    "Case": "case",
    "Clusivity": "clusivity",
    "Degree": "degree",
    "Foreign": "foreign",
    "Mood": "mood",
    "Number": "number",
    "NumType": "numeral",
    "Person": "person",
    "Polarity": "polarity",
    "Polite": "politeness",
    "PronType": "pronoun",
    "Poss": "possessive",
    "Reflex": "reflexive",
    "Tense": "tense",
    "Typo": "typo",
    "VerbForm": "form",
    "Voice": "voice",
    "Yes": "yes",

    # Feature Values
    "Imp": "imperfective",
    "Perf": "perfective",
    "Prog": "progressive",
    "Abl": "ablative",
    "Acc": "accustative",
    "Cmp": "comparative",
    "Com": "comitative",
    "Dat": "dative",
    "Gen": "genitive",
    "Ins": "instrumental",
    "Lat": "lative (denotes movement towards/to/into/onto something)",
    "Loc": "locative",
    "Nom": "nominative",
    "Ex": "exclusive",
    "In": "inclusive",
    "Pos": "positive",
    "Cnd": "conditional",
    "Ind": "indicative",
    "Sub": "subjunctive",
    "Opt": "optative",
    "Des": "desiderative",
    "Plur": "plural",
    "Sing": "singular",
    "Card": "cardinal",
    "Frac": "fractional",
    "Mult": "multiplicative",
    "Ord": "ordinal",
    "Sets": "sets",
    "1": "first",
    "2": "second",
    "3": "third",
    "Neg": "negative",
    "Elev": "elevated",
    "Dem": "demonstrative",
    "Int": "interrogative",
    "Prs": "personal",
    "Tot": "total",
    "Fut": "future",
    "Past": "past",
    "Pres": "present",
    "Conv": "converb",
    "Fin": "finite",
    "Inf": "infinitive",
    "Part": "participle",
    "Vnoun": "verbal noun",
    "Act": "active",
    "Cau": "causative",
    "Pass": "passive",
    "Rcp": "reciprocal",
    "_":" "
}

abbrev_map_ctn = {
    **COMMON_ABBREV_MAP,

    # Universal dependency relations used in Chintang.
    "orphan": "orphan relation in ellipsis",
    "reparandum": "overridden disfluency",

    # Chintang relation subtypes.
    "acl:nmlz": "clausal nominalization",
    "advcl:cntf": "counterfactual comparison adverbial clause",
    "advcl:coord": "coordinated action adverbial clause",
    "advcl:emph": "emphasizing adverbial clause",
    "advcl:purp": "purposive adverbial clause",
    "advcl:sim": "simultaneous action adverbial clause",
    "advmod:cop": "copular use of adverbial modifier",
    "advmod:emph": "emphasizing word; intensifier",
    "advmod:nmlz": "adverbial nominalization",
    "amod:nmlz": "adjectival nominalization",
    "compound:lvc": "light verb construction",
    "det:nmlz": "determiner nominalization",
    "flat:foreign": "foreign words",
    "nmod:nmlz": "nominal nominalization",
    "nsubj:outer": "outer clause nominal subject",
    "obj:caus": "agentive object in causative construction",
    "xcomp:desid": "open clausal desiderative complement",

    # Chintang features.
    "AdvType": "adverb type",
    "Animacy": "animacy",
    "Clusivity[p]": "patient clusivity",
    "Clusivity[psor]": "possessor clusivity",
    "ConvType": "converb type",
    "Deixis": "deixis",
    "Evident": "evidentiality",
    "InfStruct": "information structure",
    "Number[p]": "patient number",
    "Number[psor]": "possessor number",
    "Person[p]": "patient person",
    "Person[psor]": "possessor person",
    "Reach": "reach",
    "Red": "reduplication",

    # Chintang feature values.
    "Abs": "absolutive",
    "Abs,Erg": "absolutive; ergative",
    "AbsErg": "absolutive and ergative",
    "Access": "access",
    "Cau": "finalis case; causative",
    "CauRcp": "causative reciprocal",
    "CauRefl": "causative reflexive",
    "Cntf": "counterfactual",
    "ComplImp": "completive imperfective",
    "ComplPerf": "completive perfect",
    "ComplPerfv": "completive perfective",
    "Coord": "coordinated action",
    "Dim": "diminutive",
    "Dual": "dual",
    "Erg": "ergative",
    "Ext": "extension adverb",
    "Foc": "focus",
    "Hum": "human",
    "Imp": "imperfective; imperative",
    "Ind": "indicative; indefinite",
    "Loc": "locative; locative adverb",
    "LocCom": "locative and comitative",
    "LocErg": "locative and ergative",
    "LocLoc": "locative and locative",
    "LocLocErg": "locative, locative, and ergative",
    "LocLocPer": "locative, locative, and perlative",
    "LocPer": "locative and perlative",
    "LocPerErg": "locative, perlative, and ergative",
    "Man": "manner adverb",
    "Med": "medial",
    "Nfh": "non-firsthand",
    "Nhum": "non-human",
    "Per": "perlative",
    "PerErg": "perlative and ergative",
    "Perfv": "perfective",
    "Pos": "positive polarity; positive degree",
    "Prox": "proximal",
    "Purp": "purposive",
    "Qua": "quantifier adverb",
    "Refl": "reflexive",
    "Rel": "relative",
    "Remote": "remote",
    "Remt": "remote",
    "Top": "topic",
    "Uniq": "unique delimiter",
}

abbrev_map_xcl = {
    **COMMON_ABBREV_MAP,

    # Universal dependency relations used in Classical Armenian.
    "dislocated": "dislocated element",
    "orphan": "orphan relation in ellipsis",

    # Classical Armenian relation subtypes.
    "aux:caus": "causative auxiliary",
    "compound:redup": "reduplicated compound",
    "csubj:caus": "causative clausal subject",
    "csubj:pass": "passive clausal subject",
    "nsubj:caus": "causative nominal subject",
    "obl:agent": "agent modifier",
    "obl:arg": "oblique argument",

    # Classical Armenian features.
    "Animacy": "animacy",
    "Connegative": "connegative form",
    "Definite": "definiteness",
    "Deixis": "relative location encoded in demonstratives",
    "ExtPos": "external part of speech",

    # Classical Armenian feature values.
    "Anim": "animate",
    "Art": "article",
    "CauPass": "causative passive",
    "Def": "definite",
    "Dist": "distributive numeral",
    "Imp": "imperfective; imperative",
    "Inan": "inanimate",
    "Ind": "indicative; indefinite",
    "Med": "medial",
    "Prox": "proximal",
    "Rel": "relative",
    "Remt": "remote",
    "Spec": "specific indefinite",
}



def postorder(node, out):
    """Traverse  and collect nodes in postorder: children first, then parent."""
    for child in node.children:
        postorder(child, out)
    out.append(node)

def get_head_dependent_steps(root):
    """
    For a dependency tree rooted at `root`,
    return a list of (head, dependents) tuples,
    starting from the leaves and going up to the root.
    """
    # 1 get the nodes in postorder
    nodes_in_postorder = []
    postorder(root, nodes_in_postorder)#Collect nodes in postorder: children first, then parent.
    #print(nodes_in_postorder)

    steps = []
    # 2 start from leaves and go up
    for node in nodes_in_postorder:
        if node.children == []:# if the node has no children (is a leaf)
            continue #skip this leaf, but continue with the next node
        # if is not a leaf, get the head and its dependent
        steps.append((node, node.children))
        
    return steps

def get_subtree_nodes(node):
    """
    Return all nodes in the subtree rooted at `node`,
    sorted by token id.
    """
    nodes = []
    def dfs(n):
        nodes.append(n)
        for ch in n.children:
            dfs(ch)
    dfs(node)
    return sorted(nodes, key=lambda n: n.token["id"])






### Helpers for supported languages

def get_abbrev_map(language_code):
    abbrev_maps = {
        "ctn": abbrev_map_ctn,
        "xcl": abbrev_map_xcl,
    }
    return abbrev_maps[language_code]

def get_word_text(token, language_code):
    if language_code == "xcl":
        return (token.get("misc") or {}).get("Translit", token["form"])
    return token["form"]

def get_lemma_text(token, language_code):
    if language_code == "xcl":
        return (token.get("misc") or {}).get("LTranslit", token.get("lemma", "_"))
    return token.get("lemma", "_")

def get_gloss_text(token):
    return (token.get("misc") or {}).get("Gloss") or " "

def get_translation_text(metadata, language_code):
    translation_keys = {
        "ctn": "english",
        "xcl": "translated_text",
    }
    for key in [translation_keys[language_code], "english", "translated_text"]:
        if key in metadata:
            return metadata[key]
    return " "

GRAMMAR_RULE_PATHS = {
    "ctn": PROJECT_ROOT / "gram_rules" / "gram_rules_ctn.json",
    "xcl": PROJECT_ROOT / "gram_rules" / "gram_rules_xcl.json",
}

_GRAMMAR_RULE_CACHE = {}

def load_grammar_rules(language_code):
    """
    Load the UD grammar-rule JSON for a language.

    The rule files are expected to exist. Results are cached because one
    sentence can call the matching helpers many times.
    """
    if language_code in _GRAMMAR_RULE_CACHE:
        return _GRAMMAR_RULE_CACHE[language_code]

    import json
    with open(GRAMMAR_RULE_PATHS[language_code], "r", encoding="utf-8") as f:
        rules = json.load(f)
    _GRAMMAR_RULE_CACHE[language_code] = rules
    return rules

def _match_feats(token_feats, rule_feats):
    """
    Count matched feature constraints.

    Example:
        token_feats = {"Case": "Erg", "Number": "Sing"}
        rule_feats = {"Case": ["Erg"], "Number": ["Sing", "Plur"]}

    Returns (matched, total). The value ["*"] means any non-empty value for
    that feature is acceptable.
    """
    matched = 0
    total = 0
    token_feats = token_feats or {}
    for feat_name, allowed_values in (rule_feats or {}).items():
        if not isinstance(allowed_values, list):
            allowed_values = [allowed_values]
        total += 1
        if feat_name in token_feats and ("*" in allowed_values or token_feats[feat_name] in allowed_values):
            matched += 1
    return matched, total

def _count(condition):
    """Return 1 when a single trigger condition is true, otherwise 0."""
    return 1 if condition else 0

def score_word_rule(rule, node, language_code):
    """
    Score a rule against one conllu TokenTree node.

    This is used by explain_word(). It checks only word-level triggers:
    - upos: token["upos"]
    - lemma: get_lemma_text(token, language_code)
    - feats: token["feats"]

    Returns (matched, total). If matched == total, the rule is fully triggered.
    If matched > 0 but matched < total, it is a partial fallback candidate.
    """
    triggers = rule["triggers"]
    if "deprel" in triggers:
        return 0, 0

    token = node.token
    matched = 0
    total = 0

    if "upos" in triggers:
        total += 1
        matched += _count(token["upos"] in triggers["upos"])
    if "lemma" in triggers:
        total += 1
        matched += _count(get_lemma_text(token, language_code) in triggers["lemma"])
    if "feats" in triggers:
        feat_matched, feat_total = _match_feats(token.get("feats"), triggers["feats"])
        matched += feat_matched
        total += feat_total

    return matched, total

def _score_node_side(prefix, triggers, node, language_code):
    """
    Score head_* or dependent_* constraints against one side of a relation.

    prefix is either "head" or "dependent".
    """
    token = node.token
    matched = 0
    total = 0

    upos_key = f"{prefix}_upos"
    lemma_key = f"{prefix}_lemma"
    feats_key = f"{prefix}_feats"

    if upos_key in triggers:
        total += 1
        matched += _count(token["upos"] in triggers[upos_key])
    if lemma_key in triggers:
        total += 1
        matched += _count(get_lemma_text(token, language_code) in triggers[lemma_key])
    if feats_key in triggers:
        feat_matched, feat_total = _match_feats(token.get("feats"), triggers[feats_key])
        matched += feat_matched
        total += feat_total

    return matched, total

def score_relation_rule(rule, head, dep, language_code):
    """
    Score a rule against one dependency edge: head -> dep.

    This is used by explain_syntactic_relations(). It checks:
    - deprel: dep.token["deprel"]
    - lemma / upos / feats: either head or dependent may match
    - head_*: only the head node may match
    - dependent_* or dep_*: only the dependent node may match

    Returns (matched, total), with the same full-match / partial-fallback
    interpretation as score_word_rule().
    """
    triggers = dict(rule["triggers"])
    if "dep_upos" in triggers:
        triggers["dependent_upos"] = triggers.pop("dep_upos")
    if "dep_lemma" in triggers:
        triggers["dependent_lemma"] = triggers.pop("dep_lemma")
    if "dep_feats" in triggers:
        triggers["dependent_feats"] = triggers.pop("dep_feats")

    relation_anchor_keys = {
        "deprel",
        "head_upos", "head_lemma", "head_feats",
        "dependent_upos", "dependent_lemma", "dependent_feats",
    }
    if not any(key in triggers for key in relation_anchor_keys):
        return 0, 0

    matched = 0
    total = 0

    if "deprel" in triggers:
        total += 1
        matched += _count(dep.token["deprel"] in triggers["deprel"])
    if "lemma" in triggers:
        total += 1
        matched += _count(
            get_lemma_text(head.token, language_code) in triggers["lemma"]
            or get_lemma_text(dep.token, language_code) in triggers["lemma"]
        )
    if "upos" in triggers:
        total += 1
        matched += _count(head.token["upos"] in triggers["upos"] or dep.token["upos"] in triggers["upos"])
    if "feats" in triggers:
        head_matched, feat_total = _match_feats(head.token.get("feats"), triggers["feats"])
        dep_matched, _ = _match_feats(dep.token.get("feats"), triggers["feats"])
        matched += max(head_matched, dep_matched)
        total += feat_total

    for prefix, node in [("head", head), ("dependent", dep)]:
        side_matched, side_total = _score_node_side(prefix, triggers, node, language_code)
        matched += side_matched
        total += side_total

    return matched, total

def choose_best_rules(scored_rules, max_rules, used_rule_ids=None, long_rule_once_threshold=400):
    """
    Choose which matching rules to insert.

    Full matches are preferred. If no rule fully matches, partial matches are
    used as a fallback. Within either group, rules that match more trigger
    fields are preferred. If used_rule_ids is given, every selected rule id is
    recorded there. Long rules are skipped if they already appeared in the
    current trace.
    """
    full_matches = [item for item in scored_rules if item[1] > 0 and item[0] == item[1]]
    partial_matches = [item for item in scored_rules if item[1] > 0 and item[0] > item[1] / 2]
    candidates = full_matches or partial_matches

    def sort_key(item):
        matched, total, rule = item
        source_url = rule.get("source", {}).get("url", "")
        source_bonus = 0 if source_url.endswith("/index.html") else 1
        return (matched, total, source_bonus)

    candidates.sort(key=sort_key, reverse=True)

    selected = []
    for _, _, rule in candidates:
        is_long_rule = (
            long_rule_once_threshold is not None
            and len(rule["rule_text"]) > long_rule_once_threshold
        )
        if used_rule_ids is not None and is_long_rule and rule["id"] in used_rule_ids:
            continue
        selected.append(rule)
        if used_rule_ids is not None:
            used_rule_ids.add(rule["id"])
        if len(selected) == max_rules:
            break
    return selected

def get_word_grammar_notes(
    node,
    language_code,
    grammar_rules,
    max_rules=3,
    used_rule_ids=None,
    long_rule_once_threshold=400,
):
    """Return the best word-level UD grammar note(s) for one token."""
    scored = []
    for rule in grammar_rules:
        matched, total = score_word_rule(rule, node, language_code)
        if matched:
            scored.append((matched, total, rule))
    selected = choose_best_rules(
        scored,
        max_rules,
        used_rule_ids,
        long_rule_once_threshold,
    )
    if not selected:
        return []
    language_name = {"ctn": "Chintang", "xcl": "Classical Armenian"}.get(language_code, language_code)
    rule_lines = "\n".join(rule["rule_text"] for rule in selected)
    return [f"According to the grammar of {language_name}:\n{rule_lines}"]

def get_relation_grammar_notes(
    head,
    dep,
    language_code,
    grammar_rules,
    max_rules=3,
    used_rule_ids=None,
    long_rule_once_threshold=400,
):
    """Return the best syntax-level UD grammar note(s) for one dependency edge."""
    scored = []
    for rule in grammar_rules:
        matched, total = score_relation_rule(rule, head, dep, language_code)
        if matched:
            scored.append((matched, total, rule))
    selected = choose_best_rules(
        scored,
        max_rules,
        used_rule_ids,
        long_rule_once_threshold,
    )
    if not selected:
        return []
    language_name = {"ctn": "Chintang", "xcl": "Classical Armenian"}.get(language_code, language_code)
    rule_lines = "\n".join(rule["rule_text"] for rule in selected)
    return [f"According to the grammar of {language_name}:\n{rule_lines}"]

def explain_word(
    node,
    language_code,
    grammar_rules=None,
    max_grammar_rules=3,
    used_rule_ids=None,
    long_rule_once_threshold=400,
):
    """
    Generate explanation for a single node word, including its POS, lemma, features.
    """
    abbrev_map = get_abbrev_map(language_code)
    token = node.token
    word = get_word_text(token, language_code)
    lemma = get_lemma_text(token, language_code)
    gloss = get_gloss_text(token)

    text = ''
    text += f"The word '{word}' is a {abbrev_map[token['upos']]}. "
    if word != lemma and lemma != '_':
        text += f"Its lemma form is '{lemma}', which means [{clean_string(gloss)}]. "
    if token.get('misc') and 'MSeg' in token.get('misc'):
        text += f"'{word}' is morphologically segmented as '{token['misc']['MSeg']}'."
    if token['feats']:# if has features, explain the features
        text += f"'{word}' is"
        if 'Case' in token['feats'].keys():
            k = 'Case'
            v = token['feats'][k]
            text += f" {abbrev_map[v]} {abbrev_map[k]}"
        for k,v in token['feats'].items():
            if k == 'Case':
                continue
            if v == 'Yes':
                text += f" {abbrev_map[k]}"
            else:
                text += f" {abbrev_map[v]} {abbrev_map[k]}"
        text += f"."
    if grammar_rules:
        for note in get_word_grammar_notes(
            node,
            language_code,
            grammar_rules,
            max_grammar_rules,
            used_rule_ids,
            long_rule_once_threshold,
        ):
            text += f"\n{note}"

    if token['upos'] in ['ADP'] and not token['feats']:# if does not have features but is adposition, should have one-line explanation
        text += f"\n - So '{word}' means: [{gloss}]"

    if token['upos'] not in ['PUNCT','ADP']:# everything that is not punctuation and adposition should have one-line explanation
        text += f"\n - So '{word}' means: [{gloss}]"
        
    return text

def explain_syntactic_relations(
    head,
    dep,
    language_code,
    grammar_rules=None,
    max_grammar_rules=3,
    used_rule_ids=None,
    long_rule_once_threshold=400,
):
    """
    Generate explanation for the syntactic relation between head and dependent
    The reasoning is in the format of:
    directionality + POS -> syntactic relation type
    """
    abbrev_map = get_abbrev_map(language_code)
    head_word = get_word_text(head.token, language_code)
    dep_word = get_word_text(dep.token, language_code)
    text = ''
    subtree_nodes = get_subtree_nodes(dep)
    subtree_nodes_text = ' '.join([get_word_text(sub.token, language_code) for sub in subtree_nodes])
    if grammar_rules:
        for note in get_relation_grammar_notes(
            head,
            dep,
            language_code,
            grammar_rules,
            max_grammar_rules,
            used_rule_ids,
            long_rule_once_threshold,
        ):
            text += f"{note}"
    if dep.token['id'] < head.token['id']:
        text += f"\nAs the {abbrev_map[dep.token['upos']]} '{dep_word}' precedes the {abbrev_map[head.token['upos']]} '{head_word}', "
    else:
        text += f"\nAs the {abbrev_map[dep.token['upos']]} '{dep_word}' follows the {abbrev_map[head.token['upos']]} '{head_word}', "
    text += f"the syntactic relationship here is: '{subtree_nodes_text}' is the {abbrev_map[dep.token['deprel']]} of '{head_word}'."
    return text

def get_reasoning_steps_with_rule_ids(
    root,
    language_code,
    include_grammar_rules=True,
    max_grammar_rules=3,
    long_rule_once_threshold=400,
    number_placeholders=False,
    mask_final_translation_placeholder=False,
):
    """
    Produce reasoning steps and the set of grammar-rule ids used in the trace.

    The ids are collected only after rule matching, max_grammar_rules, and
    long-rule-once filtering have been applied.
    """
    output = ''
    grammar_rules = load_grammar_rules(language_code) if include_grammar_rules else []
    used_rule_ids = set()
    head_dependent_steps = get_head_dependent_steps(root)
    if not head_dependent_steps: # for sentences that is exactly one token, which does not have children.
        output += f"\nStep 1:"
        output += f"\n{explain_word(root, language_code, grammar_rules, max_grammar_rules, used_rule_ids, long_rule_once_threshold)}"
        output += f"\nPutting all these pieces together, the whole sentence '{get_word_text(root.token, language_code)}' translates to: '{get_translation_text(root.metadata, language_code)}'"
        output = apply_reasoning_output_options(
            output,
            number_placeholders,
            mask_final_translation_placeholder,
        )
        return output, used_rule_ids
    for i, (head, dependents) in enumerate(head_dependent_steps,1):
        output += f"\nStep {i}:"
        output += f"\n{explain_word(head, language_code, grammar_rules, max_grammar_rules, used_rule_ids, long_rule_once_threshold)}"
        for dep in dependents:
            if dep.children == []:
                output += f"\n{explain_word(dep, language_code, grammar_rules, max_grammar_rules, used_rule_ids, long_rule_once_threshold)}"
            output += f"\n{explain_syntactic_relations(head, dep, language_code, grammar_rules, max_grammar_rules, used_rule_ids, long_rule_once_threshold)}"
        #if not the last step, print phrasal translation
        if i < len(head_dependent_steps):
            output += f"\n - So '{' '.join([get_word_text(sub.token, language_code) for sub in get_subtree_nodes(head)])}' means: [Phrasal Translation]"
    output += f"\nPutting all these pieces together, the whole sentence '{' '.join([get_word_text(sub.token, language_code) for sub in get_subtree_nodes(root)])}' translates to: '{get_translation_text(root.metadata, language_code)}'"
    output = apply_reasoning_output_options(
        output,
        number_placeholders,
        mask_final_translation_placeholder,
    )
    return output, used_rule_ids

def get_reasoning_steps(
    root,
    language_code,
    include_grammar_rules=True,
    max_grammar_rules=3,
    long_rule_once_threshold=400,
    number_placeholders=False,
    mask_final_translation_placeholder=False,
):
    """
    Putting the pieces together, produce the reasoning steps for the entire tree rooted at `root`
    """
    output, _ = get_reasoning_steps_with_rule_ids(
        root,
        language_code,
        include_grammar_rules,
        max_grammar_rules,
        long_rule_once_threshold,
        number_placeholders,
        mask_final_translation_placeholder,
    )
    return output

def generate_prompts_for_llm_fill_in_place_holders(trees, language_code):
    import json
    # generate prompts for llm_fill_in_place_holders
    dict_paths = {
        "ctn": PROJECT_ROOT / "dicts_generated_from_UD" / "dict_UD_gloss_ctn.json",
        "xcl": PROJECT_ROOT / "dicts_generated_from_UD" / "dict_UD_gloss_xcl.json",
    }
    with open(dict_paths[language_code], "r", encoding="utf-8") as f:
        lemma_dict = json.load(f)

    prompts = []
    for root in trees: #[0:10]: # select number of sentences
        prompt_dict = {}
        prompt_dict['sent_id'] = root.metadata['sent_id']
        if language_code == "xcl":
            prompt_dict['text'] = root.metadata['transliterated_text']
        else:
            prompt_dict['text'] = root.metadata['text']
        prompt_dict['translation'] = get_translation_text(root.metadata, language_code)

        lemmas = []
        for node in get_subtree_nodes(root):
            if language_code == "xcl":
                lemma = (node.token.get("misc") or {}).get("LTranslit", "_")
            else:
                lemma = node.token.get("lemma", "_")
            if lemma != "_" and lemma not in lemmas:
                lemmas.append(lemma)

        wordbyword = '\n'.join([f"{lemma}: {lemma_dict.get(lemma, 'not found in dictionary')}" for lemma in lemmas])
        reasoning_step = get_reasoning_steps(root, language_code)
        prompt = f"""You are given dictionary entries for each individual word in a sentence, and a step-by-step reasoning process that explains how to combine the meanings of these individual words to phrases, and finally arrive at the meaning of the whole sentence. 
    Your task is to complete the reasoning steps below by filling in every placeholder enclosed in square brackets, as in [Phrasal Translation].
    In [Phrasal Translation], you should fill in the English translation of the complete phrase or clause formed at that step.

    Instructions:
    
    1. The dictionary entry may contain multiple possible meanings for a word, the specific lexical meaning in the context and its morphological features are already explained in the reasoning steps.
    2. Based on the explanations of each word, and the final English translation of the whole sentence (provided at the end of the reasoning steps), you task is to fill in each [Phrasal Translation] combining the meanings of the words according to the syntactic relations explained in the reasoning steps.
    3. Ensure consistency between the Phrasal Translation and the final English translation of the whole sentence.
    4. Preserve all original formatting, keep all square brackets exactly as they appear.
    5. Do not add, remove, or modify any text outside of filling in the placeholders of [Phrasal Translation].
    6. Return only the completed reasoning steps. Do not include explanations, comments, or additional text.

    Dictionary entries:
    {wordbyword}

    Reasoning steps with placeholders:
    {reasoning_step}
        """
        prompt_dict['prompt'] = prompt
        prompts.append(prompt_dict)
    
    return prompts

def load_ud_trees(folder_path):
    from pathlib import Path
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

def filter_trees_by_max_word_len(trees, language_code, max_word_len=30):
    if max_word_len is None:
        return trees

    if max_word_len < 1:
        raise ValueError("max_word_len must be at least 1")

    filtered_trees = []
    for root in trees:
        if language_code == "xcl":
            text = root.metadata.get("transliterated_text", root.metadata.get("text", ""))
        elif language_code == "ctn":
            text = root.metadata.get("text", "")
        else:
            raise ValueError(f"Unsupported language_code: {language_code}")

        if len(text.split()) <= max_word_len:
            filtered_trees.append(root)

    print(f"Filtered trees by max_word_len={max_word_len}: {len(filtered_trees)}/{len(trees)} kept.")
    return filtered_trees


def save_prompts_for_llm_fill_in_place_holders(folder_path, language_code):
    import json
    from pathlib import Path

    trees, _ = load_ud_trees(folder_path)
    trees = filter_trees_by_max_word_len(trees, language_code)
    prompts = generate_prompts_for_llm_fill_in_place_holders(trees, language_code)

    output_path = Path(f"llm_fill_in_placeholders_prompts_{language_code}.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    return output_path
