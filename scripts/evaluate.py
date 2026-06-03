import json
import argparse
import os
import glob
import csv
import re
import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU, CHRF
from sentence_transformers import SentenceTransformer

def get_index_label(i):
    """Generates Excel-style column indices: A, B, ..., Z, AA, AB, etc."""
    label = ""
    while i >= 0:
        label = chr(65 + (i % 26)) + label
        i = i // 26 - 1
    return label

def evaluate_single_file(file_path, model, bleu, chrf):
    """Evaluates a single JSON file and returns a dictionary of scores."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preds = []
    refs = []
    non_empty_preds = []
    non_empty_refs = []
    empty_count = 0

    for item in data:
        pred = item.get("final_translation") or ""
        ref = item.get("gold_translation") or ""

        preds.append(pred)
        refs.append(ref)

        if pred.strip() == "":
            empty_count += 1
        else:
            non_empty_preds.append(pred)
            non_empty_refs.append(ref)

    total_pairs = len(preds)

    # Calculate scores for ALL translations
    bleu_score_all = bleu.corpus_score(preds, [refs]).score
    chrf_score_all = chrf.corpus_score(preds, [refs]).score

    pred_embeddings_all = model.encode(preds, convert_to_tensor=True)
    ref_embeddings_all = model.encode(refs, convert_to_tensor=True)
    cosine_scores_all = F.cosine_similarity(pred_embeddings_all, ref_embeddings_all)
    sbert_score_all = cosine_scores_all.mean().item() * 100

    # Calculate metrics for NON-EMPTY translations
    if len(non_empty_preds) > 0:
        bleu_score_ne = bleu.corpus_score(non_empty_preds, [non_empty_refs]).score
        chrf_score_ne = chrf.corpus_score(non_empty_preds, [non_empty_refs]).score
        
        pred_embeddings_ne = model.encode(non_empty_preds, convert_to_tensor=True)
        ref_embeddings_ne = model.encode(non_empty_refs, convert_to_tensor=True)
        cosine_scores_ne = F.cosine_similarity(pred_embeddings_ne, ref_embeddings_ne)
        sbert_score_ne = cosine_scores_ne.mean().item() * 100
    else:
        bleu_score_ne = chrf_score_ne = sbert_score_ne = 0.0
    
    return {
        "total": total_pairs,
        "empty_count": empty_count,
        "bleu_all": bleu_score_all,
        "chrf_all": chrf_score_all,
        "sbert_all": sbert_score_all,
        "bleu_ne": bleu_score_ne,
        "chrf_ne": chrf_score_ne,
        "sbert_ne": sbert_score_ne
    }

def get_short_setting_name(file_path):
    """Maps a result filename to a short setting label."""
    filename = os.path.splitext(os.path.basename(file_path))[0]
    parts = []

    if "rft" in filename or "RFT" in filename:
        parts.append("rft")
    elif "base" in filename:
        # parts.append("no_sft_baseline")
        if "icl" in filename:
            parts.append(filename.split("_")[1])  # e.g. "gemma-4-31B-it"
            parts.append("+reasoning")
        else:
            parts.append(filename.split("_")[1])  # e.g. "gemma-4-31B-it"
    elif "no_thinking" in filename:
        parts.append("no_reasoning_baseline")
    else:
        parts.append("sft")

    checkpoint_match = re.search(r"(?:checkpoint|ckpt)[-_](\d+)", filename)
    if checkpoint_match:
        parts.append(f"ckpt{checkpoint_match.group(1)}")

    return "_".join(parts) if parts else os.path.basename(file_path)

def load_llm_judge_means(folder_path):
    """Loads LLM-as-a-judge means from the llm_judge summary CSV."""
    llm_judge_dir = os.path.join(folder_path, "llm_judge")
    summary_csv = os.path.join(llm_judge_dir, "llm_judge_summary.csv")
    if not os.path.exists(summary_csv):
        csv_files = glob.glob(os.path.join(llm_judge_dir, "*.csv"))
        if not csv_files:
            print(f"No LLM judge CSV found in: {llm_judge_dir}")
            return {}
        summary_csv = csv_files[0]

    means = {}
    with open(summary_csv, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_path = row.get("file")
            mean = row.get("mean")
            if not file_path or mean in (None, ""):
                continue
            normalized_path = os.path.abspath(os.path.normpath(file_path))
            means[normalized_path] = float(mean)
            means[os.path.basename(file_path)] = float(mean)

    print(f"Loaded LLM judge means from: {summary_csv}")
    return means


def get_llm_judge_mean(file_path, llm_judge_means):
    if not llm_judge_means:
        return None
    normalized_path = os.path.abspath(os.path.normpath(file_path))
    return llm_judge_means.get(normalized_path) or llm_judge_means.get(os.path.basename(file_path))


def write_results_csv(output_path, file_mapping, results, include_llm_judge=False, llm_judge_means=None):
    """Writes the evaluation summary in a wide CSV format for Excel."""
    header = [
        "Setting",
        "JSON file",
        "BLEU",
        "chrF",
        "SBERT",
    ]
    if include_llm_judge:
        header.append("LLMaJ")
    header.extend([
        "",
        "BLEU non-empty",
        "chrF non-empty",
        "SBERT non-empty",
        "Non-empty percentage",
    ])

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for idx, path in file_mapping.items():
            res = results[idx]
            valid_sentences = res["total"] - res["empty_count"]
            non_empty_percentage = (
                f"{valid_sentences}/{res['total']} ({((valid_sentences / res['total']) * 100):.1f}%)"
                if res["total"] > 0 else "0/0 (0.0%)"
            )

            row = [
                get_short_setting_name(path),
                os.path.basename(path),
                f"{res['bleu_all']:.2f}",
                f"{res['chrf_all']:.2f}",
                f"{res['sbert_all']:.2f}",
            ]
            if include_llm_judge:
                llm_judge_mean = get_llm_judge_mean(path, llm_judge_means or {})
                row.append(f"{llm_judge_mean:.2f}" if llm_judge_mean is not None else "")
            row.extend([
                "",
                f"{res['bleu_ne']:.2f}",
                f"{res['chrf_ne']:.2f}",
                f"{res['sbert_ne']:.2f}",
                non_empty_percentage,
            ])
            writer.writerow(row)


def main(folder_path, include_llm_judge=False):
    # Find all JSON files in the given directory
    json_files = sorted(
        glob.glob(os.path.join(folder_path, "*.json")),
        key=lambda path: (get_short_setting_name(path), os.path.basename(path).lower())
    ) # Sort by short setting name first, then by original filename
    
    if not json_files:
        print(f"No .json files found in the directory: {folder_path}")
        return

    # Setup metrics ONE TIME to save computation
    print("Initializing metrics and loading SBERT model (all-MiniLM-L6-v2)...")
    bleu = BLEU(lowercase=True, effective_order=False, tokenize='13a', smooth_method='exp')
    chrf = CHRF()
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    results = {}
    file_mapping = {}
    llm_judge_means = load_llm_judge_means(folder_path) if include_llm_judge else {}

    # Evaluate all files
    print(f"\nProcessing {len(json_files)} files...")
    used_labels = {}
    for file_path in json_files:
        idx = get_short_setting_name(file_path)
        if idx in used_labels:
            used_labels[idx] += 1
            idx = f"{idx}_{used_labels[idx]}"
        else:
            used_labels[idx] = 1

        file_mapping[idx] = file_path
        print(f"[{idx}] Evaluating {os.path.basename(file_path)}...")
        results[idx] = evaluate_single_file(file_path, sbert_model, bleu, chrf)

    # Print File Index Mapping
    print("\n" + "="*75)
    print("FILE INDEX MAPPING")
    for idx, path in file_mapping.items():
        print(f"[{idx}] {os.path.basename(path)}")

    # --- MODIFICATION START: Split into two separate tables ---
    
    # TABLE 1: All Translations
    print("\n" + "="*60)
    if include_llm_judge:
        header1 = f"{'Setting':<12} | {'BLEU':<8} | {'chrF':<8} | {'SBERT':<8} | {'LLM judge':<9}"
    else:
        header1 = f"{'Setting':<12} | {'BLEU':<8} | {'chrF':<8} | {'SBERT':<8}"
    print(header1)
    print("-" * len(header1))

    for idx in file_mapping.keys():
        res = results[idx]
        row1 = f"{idx:<12} | {res['bleu_all']:<8.2f} | {res['chrf_all']:<8.2f} | {res['sbert_all']:<8.2f}"
        if include_llm_judge:
            llm_judge_mean = get_llm_judge_mean(file_mapping[idx], llm_judge_means)
            llm_judge_text = f"{llm_judge_mean:.2f}" if llm_judge_mean is not None else ""
            row1 += f" | {llm_judge_text:<9}"
        print(row1)

    # TABLE 2: Non-Empty Only
    print("\nNON-EMPTY RESULTS:")
    header2 = f"{'Setting':<12} | {'BLEU':<8} | {'chrF':<8} | {'SBERT':<8} | {'None-empty percentage':<8}"
    print(header2)
    print("-" * len(header2))

    for idx in file_mapping.keys():
        res = results[idx]
        valid_sentences = res['total'] - res['empty_count']
        non_empty_rate = f"{valid_sentences}/{res['total']} ({((valid_sentences / res['total']) * 100):.1f}%)"
        
        row2 = (
            f"{idx:<12} | "
            f"{res['bleu_ne']:<8.2f} | {res['chrf_ne']:<8.2f} | {res['sbert_ne']:<8.2f}"
            f"| {non_empty_rate:<15}"
        )
        print(row2)
        
    print("="*60 + "\n")
    # --- MODIFICATION END ---

    model_name = os.path.basename(os.path.normpath(folder_path))
    output_csv = os.path.join(folder_path, f"{model_name}.csv")
    write_results_csv(output_csv, file_mapping, results, include_llm_judge, llm_judge_means)
    print(f"CSV summary written to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple machine translation JSON files in a directory.")
    parser.add_argument("folder", help="Path to the directory containing .json evaluation files")
    parser.add_argument(
        "--include_llm_judge",
        action="store_true",
        help="Include mean LLM judge scores from <folder>/llm_judge/*.csv.",
    )
    args = parser.parse_args()
    
    main(args.folder, args.include_llm_judge)
