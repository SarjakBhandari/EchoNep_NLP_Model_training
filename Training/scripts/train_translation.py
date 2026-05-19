from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "translation_model"
DEFAULT_BASE_MODEL = "../models/nllb"

LANGUAGE_MAP = {
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "english": "eng_Latn",
    "ne": "npi_Deva",
    "np": "npi_Deva",
    "nep": "npi_Deva",
    "nepali": "npi_Deva",
}

HEADER_SOURCE_NAMES = {
    "source", "source_text", "text_a", "input", "english", "en", "from",
}
HEADER_TARGET_NAMES = {
    "target", "target_text", "text_b", "output", "nepali", "ne", "to",
}
HEADER_KEYWORDS = HEADER_SOURCE_NAMES | HEADER_TARGET_NAMES | {"category", "lang", "language"}
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u200b", "").replace("\ufeff", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def script_counts(text: str) -> Tuple[int, int]:
    devanagari = len(DEVANAGARI_PATTERN.findall(text))
    latin = len(LATIN_PATTERN.findall(text))
    return devanagari, latin


def guess_language(text: str) -> Optional[str]:
    devanagari, latin = script_counts(text)
    if devanagari == 0 and latin == 0:
        return None
    if devanagari >= max(latin * 2, 2):
        return "ne"
    if latin >= max(devanagari * 2, 2):
        return "en"
    return None


def canonical_lang_code(lang: str) -> str:
    return LANGUAGE_MAP.get(lang.lower().strip(), lang.strip())


def ensure_lang_code(lang: str) -> str:
    return canonical_lang_code(lang)


def should_skip_pair(source_text: str, target_text: str) -> bool:
    return not (source_text and target_text)


def iter_tabular_files(data_dir: Path, include_hidden: bool = False) -> Iterator[Path]:
    excluded_dirs = {"external", "processed", "raw", "checkpoints", "venv", ".git"}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        if not include_hidden and any(part.startswith(".") for part in path.parts):
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        yield path


def looks_like_header(row: Sequence[str]) -> bool:
    cells = [normalize_text(cell).lower() for cell in row if normalize_text(cell)]
    if len(cells) < 2:
        return False
    if any(cell in HEADER_KEYWORDS for cell in cells):
        return True
    short_alpha_cells = [
        cell for cell in cells
        if re.fullmatch(r"[a-z_][a-z0-9_\- ]{0,30}", cell)
    ]
    return len(short_alpha_cells) == len(cells) and all(
        len(cell.split()) <= 3 for cell in cells
    )


def choose_header_columns(header: Sequence[str]) -> Tuple[int, int]:
    normalized = [normalize_text(cell).lower() for cell in header]
    source_index = next(
        (i for i, cell in enumerate(normalized) if cell in HEADER_SOURCE_NAMES), 0
    )
    target_index = next(
        (i for i, cell in enumerate(normalized)
         if cell in HEADER_TARGET_NAMES and i != source_index),
        1,
    )
    if source_index == target_index:
        target_index = 1 if source_index != 1 else 0
    return source_index, target_index


def read_tabular_file(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        lines = handle.readlines()

    if not lines:
        return []

    start_index = 0
    first_line = lines[0].strip().split("\t")
    if looks_like_header(first_line):
        start_index = 1

    for line in lines[start_index:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        source_text = normalize_text(parts[0])
        target_text = normalize_text(parts[1])
        if not source_text or not target_text:
            continue
        yield {
            "source_text": source_text,
            "target_text": target_text,
            "source_lang": "en",
            "target_lang": "ne",
            "source_file": path.name,
        }


def load_examples(data_dir: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for path in iter_tabular_files(data_dir):
        examples.extend(read_tabular_file(path))
    return examples


def deduplicate_examples(examples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for example in examples:
        source_text = normalize_text(example.get("source_text", ""))
        target_text = normalize_text(example.get("target_text", ""))
        key = (source_text, target_text)
        deduped[key] = {
            **example,
            "source_text": source_text,
            "target_text": target_text,
            "source_lang": "en",
            "target_lang": "ne",
        }
    return list(deduped.values())


def save_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_examples(rows: List[Dict[str, Any]], seed: int = 42):
    return rows, [], []   # all rows go to training


# ── tokenisation ───────────────────────────────────────────────────────────────

def tokenize_examples(
    batch: Dict[str, List[str]],
    tokenizer,
    max_source_length: int,
    max_target_length: int,
) -> Dict[str, Any]:
    """Tokenise a batch for NLLB without using the removed as_target_tokenizer()
    or the unreliable text_target= kwarg.

    Modern approach (Transformers >= 4.35):
      - Encode the source with tokenizer.src_lang = <source_lang_code>
      - Encode the target by temporarily setting tokenizer.src_lang =
        <target_lang_code>, which makes the tokenizer prepend the correct
        language-ID token for the decoder.
    """
    input_ids_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    for source_text, target_text, source_lang, target_lang in zip(
        batch["source_text"],
        batch["target_text"],
        batch["source_lang"],
        batch["target_lang"],
    ):
        src_code = ensure_lang_code(source_lang)   # e.g. "eng_Latn"
        tgt_code = ensure_lang_code(target_lang)   # e.g. "npi_Deva"

        # encode source
        tokenizer.src_lang = src_code
        enc = tokenizer(source_text, truncation=True, max_length=max_source_length)

        # encode target: swap src_lang so the tokenizer prepends the target
        # language-ID token (required by NLLB's sentencepiece vocabulary)
        tokenizer.src_lang = tgt_code
        lbl = tokenizer(target_text, truncation=True, max_length=max_target_length)

        # restore (keeps state predictable for the next sample)
        tokenizer.src_lang = src_code

        input_ids_list.append(enc["input_ids"])
        attention_mask_list.append(enc["attention_mask"])
        labels_list.append(lbl["input_ids"])

    return {
        "input_ids":      input_ids_list,
        "attention_mask": attention_mask_list,
        "labels":         labels_list,
    }


class _TokenizeFn:
    """Picklable callable for Dataset.map() worker processes.

    A lambda that closes over a tokenizer cannot be pickled by multiprocess
    on Windows, causing the worker pool to crash silently.  This named class
    is always picklable.  The tokenizer is lazy-loaded inside each worker
    from name_or_path so the heavy object is never serialised.
    """

    def __init__(
        self,
        tokenizer_path: str,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.tokenizer_path = tokenizer_path
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self._tokenizer = None   # lazy, populated inside each worker

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, use_fast=False
            )
        return self._tokenizer

    def __call__(self, batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenize_examples(
            batch,
            self._get_tokenizer(),
            self.max_source_length,
            self.max_target_length,
        )


def build_datasets(
    rows: List[Dict[str, Any]],
    tokenizer,
    max_source_length: int,
    max_target_length: int,
) -> Tuple[Dataset, Optional[Dataset]]:
    train_rows, valid_rows, test_rows = split_examples(rows)
    data_dir = PROJECT_ROOT / "data" / "processed"
    save_jsonl(data_dir / "train.jsonl", train_rows)
    save_jsonl(data_dir / "valid.jsonl", valid_rows)
    save_jsonl(data_dir / "test.jsonl", test_rows)
    save_jsonl(data_dir / "prepared_corpus.jsonl", rows)

    # tokenizer.name_or_path is set by from_pretrained() — safe to pass to workers
    tok_path: str = getattr(tokenizer, "name_or_path", None) or DEFAULT_BASE_MODEL
    tok_fn = _TokenizeFn(tok_path, max_source_length, max_target_length)

    train_dataset = Dataset.from_list(train_rows)
    train_tokenized = train_dataset.map(
        tok_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    valid_tokenized = None
    if valid_rows:
        valid_dataset = Dataset.from_list(valid_rows)
        valid_tokenized = valid_dataset.map(
            tok_fn,
            batched=True,
            remove_columns=valid_dataset.column_names,
        )

    return train_tokenized, valid_tokenized


def train_model(
    train_tokenized: Dataset,
    valid_tokenized: Optional[Dataset],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    training_cfg = config.get("training", {})
    base_model              = str(training_cfg.get("base_model", DEFAULT_BASE_MODEL))
    cpu_threads             = int(training_cfg.get("cpu_threads", 12))
    grad_accum              = int(training_cfg.get("gradient_accumulation_steps", 2))
    weight_decay            = float(training_cfg.get("weight_decay", 0.0))
    warmup_ratio            = float(training_cfg.get("warmup_ratio", 0.1))
    max_grad_norm           = float(training_cfg.get("max_grad_norm", 1.0))
    save_total_limit        = int(training_cfg.get("save_total_limit", 3))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 2))
    max_target_length       = int(training_cfg.get("max_target_length", 160))
    num_train_epochs        = int(training_cfg.get("num_train_epochs", 10))
    per_device_batch        = int(training_cfg.get("batch_size", 8))
    logging_steps           = int(training_cfg.get("logging_steps", 20))

    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    model     = AutoModelForSeq2SeqLM.from_pretrained(base_model, use_safetensors=True)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # forced_bos_token_id: tells the decoder to always start with the
    # Nepali language-ID token during generation (replaces the old
    # set_lang_for_generation / as_target_tokenizer pattern).
    # NllbTokenizer uses convert_tokens_to_ids — lang_code_to_id does not exist.
    tgt_code      = canonical_lang_code("ne")   # "npi_Deva"
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)
    # convert_tokens_to_ids returns unk_token_id when the token is missing;
    # only apply the override when we resolved a real token.
    if forced_bos_id != tokenizer.unk_token_id:
        model.config.forced_bos_token_id = forced_bos_id
    else:
        print(f"Warning: could not resolve forced_bos_token_id for '{tgt_code}'; "
              "generation may not produce the correct target language.")

    has_validation = valid_tokenized is not None and len(valid_tokenized) > 0

    try:
        train_samples = len(train_tokenized)
    except Exception:
        train_samples = 0
    steps_per_epoch   = max(1, train_samples // (per_device_batch * max(1, grad_accum)))
    total_train_steps = steps_per_epoch * max(1, num_train_epochs)
    warmup_steps      = int(total_train_steps * warmup_ratio)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        eval_strategy="epoch" if has_validation else "no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=save_total_limit,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized if has_validation else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            if has_validation else []
        ),
    )

    print(f"Training samples  : {len(train_tokenized)}")
    print(f"Validation samples: {len(valid_tokenized) if has_validation else 0}")
    print(f"CPU threads: {cpu_threads} | Grad accumulation: {grad_accum}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_train_steps}")
    print(f"Starting fine-tuning for {num_train_epochs} epochs ...")
    trainer.train()

    print(f"Saving trained model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "base_model":       base_model,
        "output_dir":       str(output_dir),
        "train_examples":   len(train_tokenized),
        "valid_examples":   len(valid_tokenized) if has_validation else 0,
        "max_target_length": max_target_length,
        "gpu_available":    torch.cuda.is_available(),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a translation model from local CSV/TSV data."
    )
    parser.add_argument("--config",       default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-dir",     default=None)
    parser.add_argument("--output-dir",   default=None)
    parser.add_argument("--base-model",   default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed",         type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()
    set_seed(args.seed)

    config       = load_config(Path(args.config))
    data_cfg     = config.get("data", {})
    training_cfg = config.get("training", {})

    data_dir   = Path(args.data_dir   or data_cfg.get("data_dir",   DEFAULT_DATA_DIR))
    output_dir = Path(args.output_dir or training_cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    base_model = args.base_model      or training_cfg.get("base_model", DEFAULT_BASE_MODEL)
    max_source_length = int(training_cfg.get("max_source_length", 160))
    max_target_length = int(training_cfg.get("max_target_length", 160))

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    examples = load_examples(data_dir)
    examples = deduplicate_examples(examples)
    if not examples:
        raise RuntimeError(f"No usable translation pairs found in {data_dir}")

    random.Random(args.seed).shuffle(examples)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(examples)} usable pairs from {data_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    train_tokenized, valid_tokenized = build_datasets(
        examples,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    prepared_summary = {
        "data_dir":     str(data_dir),
        "output_dir":   str(output_dir),
        "example_count": len(examples),
        "file_count":   len(list(iter_tabular_files(data_dir))),
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(prepared_summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(prepared_summary, ensure_ascii=False, indent=2))

    if args.prepare_only:
        print("Preparation complete. Skipping training because --prepare-only was set.")
        return

    summary = train_model(train_tokenized, valid_tokenized, config, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    main()