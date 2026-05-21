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
    python scripts/evaluate_translation.py

Defaults:
    checkpoint: checkpoints/translation_model
    src_tsv:     data/tests.tsv
    out:         reports/model_eval.json
    src_lang:    npi_Deva
    tgt_lang:    eng_Latn
"""

from __future__ import annotations
import argparse, json, logging
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

def resolve_lang_token_id(tokenizer, lang_code: str):
    if not lang_code:
        return None
    if hasattr(tokenizer, "lang_code_to_id"):
        return tokenizer.lang_code_to_id.get(lang_code)
    token_id = tokenizer.convert_tokens_to_ids(lang_code)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None:
        return None
    if unk_token_id is not None and token_id == unk_token_id:
        return None
    return token_id

def generate_batch(model, tokenizer, inputs: List[str], device: torch.device,
                   max_length: int, generation_kwargs: Dict) -> List[str]:
    enc = tokenizer(inputs, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model.generate(**enc, **generation_kwargs)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)
    logger.info("Saved report to %s", path)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translation evaluation using BLEU score.")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/translation_model"))
    p.add_argument("--src_tsv", type=Path, default=Path("data/tests.tsv"), help="TSV file with English source and Nepali reference")
    p.add_argument("--out", type=Path, default=Path("reports/model_eval.json"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--src_lang", type=str, default="npi_Deva")
    p.add_argument("--tgt_lang", type=str, default="eng_Latn")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    srcs, refs = read_tsv(args.src_tsv)
    if len(srcs) != len(refs):
        logger.warning("Source and reference counts differ: src=%d refs=%d. Truncating.", len(srcs), len(refs))
        n = min(len(srcs), len(refs))
        srcs, refs = srcs[:n], refs[:n]

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if hasattr(tokenizer, "src_lang") and args.src_lang:
        tokenizer.src_lang = args.src_lang

    model_kwargs = {"dtype": torch.float16} if device.type == "cuda" else {}
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, **model_kwargs).to(device).eval()

    generation_kwargs = {"max_length": args.max_length, "num_beams": 4, "do_sample": False}
    lang_id = resolve_lang_token_id(tokenizer, args.tgt_lang)
    if lang_id is not None:
        generation_kwargs["forced_bos_token_id"] = lang_id
    elif args.tgt_lang:
        logger.warning("Target language %s was not found in tokenizer vocabulary", args.tgt_lang)

    preds: List[str] = []
    for i in tqdm(range(0, len(srcs), args.batch_size), desc="Generating"):
        batch_src = srcs[i:i+args.batch_size]
        preds.extend(generate_batch(model, tokenizer, batch_src, device, args.max_length, generation_kwargs))

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(preds, [refs])

    # Build report
    report = {
        "total_examples": len(refs),
        "BLEU": {"score": bleu.score},
        "meta": {
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "tgt_lang": args.tgt_lang,
        }
    }

    save_json(args.out, report)

    # Print summary
    summary = {
        "total_examples": len(refs),
        "BLEU_score": bleu.score,
        "report_file": str(args.out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info(f"BLEU score: {bleu.score:.2f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
