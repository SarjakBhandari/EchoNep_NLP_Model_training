#!/usr/bin/env python3
"""
evaluate_translation.py

Translation evaluation using a model checkpoint and TSV files.

Each TSV is expected to contain two columns:
  English source sentence (column 1)
  Nepali reference translation (column 2)

Example (tab-separated):
English Nepali
Hello, my name is Raj.  नमस्ते, मेरो नाम राज हो।
How are you? Where are you from?    कस्तो छ? तिमी कहाँ देखि आएछौ?

Usage:
  python scripts/evaluate_translation.py --checkpoint checkpoints/translation_model \
    --src_tsv data/tests.tsv --out reports/model_eval.json
"""

from __future__ import annotations
import argparse, json, logging, math
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu

# Logging
logger = logging.getLogger("eval_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def read_tsv(path: Path):
    df = pd.read_csv(path, sep="\t", header=0)
    srcs = df.iloc[:,0].astype(str).tolist()
    refs = df.iloc[:,1].astype(str).tolist()
    return srcs, refs

def token_equal(a: str, b: str) -> bool:
    return a.split() == b.split()

def generate_batch(model, tokenizer, inputs: List[str], device: torch.device,
                   max_length: int, generation_kwargs: Dict) -> List[str]:
    enc = tokenizer(inputs, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model.generate(**enc, **generation_kwargs)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

def evaluate_strict(refs: List[str], preds: List[str], sample_mismatches: int = 10) -> Dict:
    n = len(refs)
    counts = {k:0 for k in ["exact_match_strict","exact_match_strict_stripped",
                            "token_exact_match","char_length_match","token_length_match"]}
    mismatches = []
    for i,(r,p) in enumerate(zip(refs,preds)):
        if r == p: counts["exact_match_strict"] += 1
        if r.strip() == p.strip(): counts["exact_match_strict_stripped"] += 1
        if token_equal(r,p): counts["token_exact_match"] += 1
        if len(r) == len(p): counts["char_length_match"] += 1
        if len(r.split()) == len(p.split()): counts["token_length_match"] += 1
        if len(mismatches) < sample_mismatches and r != p:
            mismatches.append({"index":i,"ref":r,"pred":p})
    return {
        "total_examples": n,
        "exact_match_strict": {"count": counts["exact_match_strict"], "rate": counts["exact_match_strict"]/n if n else 0.0},
        "exact_match_strict_stripped": {"count": counts["exact_match_strict_stripped"], "rate": counts["exact_match_strict_stripped"]/n if n else 0.0},
        "token_exact_match": {"count": counts["token_exact_match"], "rate": counts["token_exact_match"]/n if n else 0.0},
        "char_length_match": {"count": counts["char_length_match"], "rate": counts["char_length_match"]/n if n else 0.0},
        "token_length_match": {"count": counts["token_length_match"], "rate": counts["token_length_match"]/n if n else 0.0},
        "mismatch_sample_count": len(mismatches),
        "mismatches": mismatches,
        "note": "Strict metrics enforce case, punctuation, token order, and lengths."
    }

def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)
    logger.info("Saved report to %s", path)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translation evaluation using a model checkpoint and TSV files.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--src_tsv", type=Path, required=True, help="TSV file with English source and Nepali reference")
    p.add_argument("--out", type=Path, default=Path("reports/model_eval.json"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample", type=int, default=10)
    p.add_argument("--tgt_lang", type=str, default=None)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    srcs, refs = read_tsv(args.src_tsv)
    if len(srcs) != len(refs):
        logger.warning("Source and reference counts differ: src=%d refs=%d. Truncating.", len(srcs), len(refs))
        n = min(len(srcs), len(refs))
        srcs, refs = srcs[:n], refs[:n]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device).eval()

    generation_kwargs = {"max_length": args.max_length, "num_beams": 1, "do_sample": False, "early_stopping": True}
    if args.tgt_lang and hasattr(tokenizer,"lang_code_to_id"):
        lang_id = tokenizer.lang_code_to_id.get(args.tgt_lang)
        if lang_id is not None:
            generation_kwargs["forced_bos_token_id"] = lang_id

    preds: List[str] = []
    for i in tqdm(range(0,len(srcs),args.batch_size), desc="Generating"):
        batch_src = srcs[i:i+args.batch_size]
        preds.extend(generate_batch(model, tokenizer, batch_src, device, args.max_length, generation_kwargs))

    # Strict metrics
    report = evaluate_strict(refs, preds, sample_mismatches=args.sample)

    # Official metrics
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    report["BLEU"] = {"score": bleu.score}
    report["chrF"] = {"score": chrf.score}

    report["meta"] = {"checkpoint": str(args.checkpoint), "device": str(device),
                      "batch_size": args.batch_size, "max_length": args.max_length,
                      "tgt_lang": args.tgt_lang}

    save_json(args.out, report)

    # Print concise summary
    summary = {
        "total_examples": report["total_examples"],
        "BLEU": report["BLEU"]["score"],
        "chrF": report["chrF"]["score"],
        "exact_match_strict_rate": report["exact_match_strict"]["rate"],
        "token_exact_match_rate": report["token_exact_match"]["rate"],
        "mismatch_sample_count": report["mismatch_sample_count"],
        "report_file": str(args.out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if report["mismatches"]:
        print("\nFirst mismatches (index, ref, pred):")
        for m in report["mismatches"]:
            print(f"{m['index']}:\n  REF:  {m['ref']}\n  PRED: {m['pred']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
