from __future__ import annotations

import argparse
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
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
    set_seed,
)
import sacrebleu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "translation_model"
DEFAULT_BASE_MODEL = str(PROJECT_ROOT / "models" / "nllb")

# ── language helpers ───────────────────────────────────────────────────────────

LANGUAGE_MAP = {
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "english": "eng_Latn",
    "ne": "npi_Deva",
    "np": "npi_Deva",
    "nep": "npi_Deva",
    "nepali": "npi_Deva",
}

ALLOWED_NLLB_CODES = {"eng_Latn", "npi_Deva"}

HEADER_SOURCE_NAMES = {
    "source", "source_text", "text_a", "input", "english", "en", "from",
}
HEADER_TARGET_NAMES = {
    "target", "target_text", "text_b", "output", "nepali", "ne", "to",
}
HEADER_KEYWORDS = HEADER_SOURCE_NAMES | HEADER_TARGET_NAMES | {"category", "lang", "language"}

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")


# ── config ─────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def resolve_project_path(value: Any, default: Path) -> Path:
    path = Path(value) if value is not None else default
    return path if path.is_absolute() else PROJECT_ROOT / path


# ── text utilities ─────────────────────────────────────────────────────────────

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
    return None  # mixed / ambiguous — reject


def canonical_lang_code(lang: str) -> str:
    return LANGUAGE_MAP.get(lang.lower().strip(), lang.strip())


def ensure_lang_code(lang: str) -> str:
    return canonical_lang_code(lang)


def is_allowed_lang(code: str) -> bool:
    return canonical_lang_code(code) in ALLOWED_NLLB_CODES


# ── file discovery ─────────────────────────────────────────────────────────────

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


# ── data loading ───────────────────────────────────────────────────────────────

def read_tabular_file(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Read a TSV/CSV file and emit *bidirectional* translation pairs.
    """
    with path.open("r", encoding="utf-8-sig") as fh:
        lines = fh.readlines()

    if not lines:
        return

    start_index = 0
    first_line_parts = lines[0].strip().split("\t")
    if looks_like_header(first_line_parts):
        start_index = 1

    for line in lines[start_index:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        col0 = normalize_text(parts[0])
        col1 = normalize_text(parts[1])
        if not col0 or not col1:
            continue

        lang0 = guess_language(col0)
        lang1 = guess_language(col1)

        # Reject rows where language cannot be determined or is not EN/NE
        if lang0 not in ("en", "ne") or lang1 not in ("en", "ne"):
            continue
        # Both columns must be different languages
        if lang0 == lang1:
            continue

        # Normalise so en_text is always English, ne_text is always Nepali
        if lang0 == "en":
            en_text, ne_text = col0, col1
        else:
            en_text, ne_text = col1, col0

        base = {"source_file": path.name}

        # EN → NE
        yield {
            **base,
            "source_text": en_text,
            "target_text": ne_text,
            "source_lang": "en",
            "target_lang": "ne",
            "direction": "en2ne",
        }

        # NE → EN  (reverse pair)
        yield {
            **base,
            "source_text": ne_text,
            "target_text": en_text,
            "source_lang": "ne",
            "target_lang": "en",
            "direction": "ne2en",
        }


def load_examples(data_dir: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for path in iter_tabular_files(data_dir):
        examples.extend(read_tabular_file(path))
    return examples


def deduplicate_examples(examples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate on (source_text, target_text, direction)."""
    deduped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for example in examples:
        source_text = normalize_text(example.get("source_text", ""))
        target_text = normalize_text(example.get("target_text", ""))
        direction = example.get("direction", "en2ne")

        # Extra guard: reject pairs whose langs are not allowed
        if not is_allowed_lang(example.get("source_lang", "en")):
            continue
        if not is_allowed_lang(example.get("target_lang", "ne")):
            continue

        key = (source_text, target_text, direction)
        deduped[key] = {
            **example,
            "source_text": source_text,
            "target_text": target_text,
        }
    return list(deduped.values())


# ── persistence ────────────────────────────────────────────────────────────────

def save_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── splitting ──────────────────────────────────────────────────────────────────

def split_examples(
    rows: List[Dict[str, Any]],
    validation_split: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return [], [], []

    ratio = max(0.0, min(float(validation_split), 0.5))
    if ratio <= 0.0 or len(rows) < 10:
        return rows, [], []

    rng = random.Random(seed)

    # Group by direction
    by_direction: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        d = row.get("direction", "en2ne")
        by_direction.setdefault(d, []).append(row)

    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []

    for direction_rows in by_direction.values():
        shuffled = direction_rows[:]
        rng.shuffle(shuffled)
        valid_size = max(1, int(len(shuffled) * ratio))
        if valid_size >= len(shuffled):
            valid_size = max(1, len(shuffled) - 1)
        valid_rows.extend(shuffled[:valid_size])
        train_rows.extend(shuffled[valid_size:])

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    return train_rows, valid_rows, []


# ── tokenisation ───────────────────────────────────────────────────────────────

def tokenize_examples(
    batch: Dict[str, List[str]],
    tokenizer,
    max_source_length: int,
    max_target_length: int,
) -> Dict[str, Any]:
    input_ids_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    for source_text, target_text, source_lang, target_lang in zip(
        batch["source_text"],
        batch["target_text"],
        batch["source_lang"],
        batch["target_lang"],
    ):
        src_code = ensure_lang_code(source_lang)
        tgt_code = ensure_lang_code(target_lang)

        # Encode source
        tokenizer.src_lang = src_code
        enc = tokenizer(source_text, truncation=True, max_length=max_source_length)

        # Encode target with target-language BOS prepended
        tokenizer.src_lang = tgt_code
        lbl = tokenizer(target_text, truncation=True, max_length=max_target_length)

        # Restore to source lang for next sample
        tokenizer.src_lang = src_code

        input_ids_list.append(enc["input_ids"])
        attention_mask_list.append(enc["attention_mask"])
        labels_list.append(lbl["input_ids"])

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


class _TokenizeFn:
    """
    Picklable callable for Dataset.map() worker processes.
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
        self._tokenizer = None

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
    validation_split: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """
    Split, save, tokenise.

    Returns (train_tokenized, valid_tokenized, valid_raw).
    """
    train_rows, valid_rows, _ = split_examples(
        rows, validation_split=validation_split, seed=seed
    )

    data_dir = PROJECT_ROOT / "data" / "processed"
    save_jsonl(data_dir / "train.jsonl", train_rows)
    save_jsonl(data_dir / "valid.jsonl", valid_rows)
    save_jsonl(data_dir / "prepared_corpus.jsonl", rows)

    tok_path: str = getattr(tokenizer, "name_or_path", None) or DEFAULT_BASE_MODEL
    tok_fn = _TokenizeFn(tok_path, max_source_length, max_target_length)

    train_dataset = Dataset.from_list(train_rows)
    train_tokenized = train_dataset.map(
        tok_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    valid_tokenized: Optional[Dataset] = None
    valid_raw: Optional[Dataset] = None

    if valid_rows:
        valid_raw = Dataset.from_list(valid_rows)
        valid_tokenized = valid_raw.map(
            tok_fn,
            batched=True,
            remove_columns=valid_raw.column_names,
        )

    return train_tokenized, valid_tokenized, valid_raw


# ── metrics ────────────────────────────────────────────────────────────────────

def _safe_bleu(hypotheses: List[str], references: List[str]) -> float:
    if not hypotheses or not references:
        return 0.0
    try:
        return sacrebleu.corpus_bleu(hypotheses, [references]).score
    except Exception:
        return 0.0


def make_compute_metrics(tokenizer, valid_raw: Optional[Dataset]):
    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 (padding sentinel) before decoding labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        n = max(len(decoded_labels), 1)

        # ── overall metrics ────────────────────────────────────────────────
        exact_matches = sum(
            1 for p, r in zip(decoded_preds, decoded_labels)
            if p.strip() == r.strip()
        )
        token_overlap_total = 0.0
        for pred, ref in zip(decoded_preds, decoded_labels):
            ref_tokens = set(ref.split())
            pred_tokens = set(pred.split())
            if ref_tokens:
                token_overlap_total += len(ref_tokens & pred_tokens) / len(ref_tokens)

        overall_bleu = _safe_bleu(decoded_preds, decoded_labels)

        results: Dict[str, float] = {
            "bleu": overall_bleu,
            "exact_match": exact_matches / n,
            "token_overlap": token_overlap_total / n,
        }

        # ── per-direction metrics ──────────────────────────────────────────
        if valid_raw is not None and len(valid_raw) == len(decoded_preds):
            directions = valid_raw["direction"]

            for tag in ("en2ne", "ne2en"):
                indices = [i for i, d in enumerate(directions) if d == tag]
                if not indices:
                    results[f"bleu_{tag}"] = 0.0
                    results[f"token_overlap_{tag}"] = 0.0
                    continue

                sub_preds = [decoded_preds[i] for i in indices]
                sub_refs = [decoded_labels[i] for i in indices]

                sub_bleu = _safe_bleu(sub_preds, sub_refs)

                sub_overlap = 0.0
                for p, r in zip(sub_preds, sub_refs):
                    ref_tokens = set(r.split())
                    pred_tokens = set(p.split())
                    if ref_tokens:
                        sub_overlap += len(ref_tokens & pred_tokens) / len(ref_tokens)

                results[f"bleu_{tag}"] = sub_bleu
                results[f"token_overlap_{tag}"] = sub_overlap / max(len(indices), 1)

                print(
                    f"  [{tag}] n={len(indices):4d}  "
                    f"BLEU={sub_bleu:.2f}  "
                    f"tok_overlap={results[f'token_overlap_{tag}']:.4f}"
                )

        return results

    return compute_metrics


# ── trainer subclass ───────────────────────────────────────────────────────────

class NLLBSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Defensive trainer for NLLB/M2M models.

    - Removes decoder_inputs_embeds when decoder_input_ids is present.
    - Calls model(**filtered_inputs) directly to compute loss to avoid
      Trainer internals that may reintroduce problematic kwargs.
    """

    def _strip_decoder_inputs_embeds(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            return inputs
        if "decoder_input_ids" in inputs and "decoder_inputs_embeds" in inputs:
            return {k: v for k, v in inputs.items() if k != "decoder_inputs_embeds"}
        return inputs

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # Defensive copy / strip
        filtered_inputs = self._strip_decoder_inputs_embeds(dict(inputs))

        # Ensure tensors are on the correct device (Trainer normally handles this)
        # but model(**filtered_inputs) will accept them as-is since Trainer moved them.
        outputs = model(**filtered_inputs)

        # Prefer outputs.loss if available, else assume first element is loss
        loss = getattr(outputs, "loss", None)
        if loss is None:
            if isinstance(outputs, tuple) and len(outputs) > 0:
                loss = outputs[0]
            else:
                raise ValueError("Model did not return a loss in outputs")

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs, *args, **kwargs):
        # Strip problematic key(s) before any Trainer logic
        inputs = self._strip_decoder_inputs_embeds(dict(inputs))
        # Forward to Trainer.training_step which will call our compute_loss
        return super().training_step(model, inputs, *args, **kwargs)


# ── training ───────────────────────────────────────────────────────────────────

def train_model(
    train_tokenized: Dataset,
    valid_tokenized: Optional[Dataset],
    valid_raw: Optional[Dataset],
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
    num_train_epochs        = int(training_cfg.get("num_train_epochs", 2))
    per_device_batch        = int(training_cfg.get("batch_size", 8))
    logging_steps           = int(training_cfg.get("logging_steps", 20))
    dataloader_workers      = int(training_cfg.get("dataloader_workers", 0))
    label_smoothing_factor  = float(training_cfg.get("label_smoothing_factor", 0.1))

    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, use_safetensors=True)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Clear forced_bos_token_id because we use tokenizer.src_lang trick for labels
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.forced_bos_token_id = None
    if hasattr(model.config, "forced_bos_token_id"):
        model.config.forced_bos_token_id = None

    has_validation = valid_tokenized is not None and len(valid_tokenized) > 0

    compute_metrics = (
        make_compute_metrics(tokenizer, valid_raw) if has_validation else None
    )

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
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        label_smoothing_factor=label_smoothing_factor,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        load_best_model_at_end=has_validation,
        metric_for_best_model="bleu" if has_validation else None,
        greater_is_better=True if has_validation else None,
        fp16=bool(training_cfg.get("fp16", torch.cuda.is_available())),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        report_to="none",
        save_total_limit=save_total_limit,
        # include_inputs_for_metrics removed — dropped in Transformers 4.38+
    )

    trainer = NLLBSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized if has_validation else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            if has_validation else []
        ),
    )

    print(f"Training samples  : {len(train_tokenized)}")
    print(f"Validation samples: {len(valid_tokenized) if has_validation else 0}")
    print(f"CPU threads: {cpu_threads}  |  Grad accumulation: {grad_accum}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Steps/epoch: {steps_per_epoch}  |  Total steps: {total_train_steps}")
    print(f"Label smoothing: {label_smoothing_factor}")
    if has_validation:
        print("Validation (BLEU + per-direction breakdown) computed each epoch.")
    print(f"Starting fine-tuning for {num_train_epochs} epochs …")

    trainer.train()

    print(f"Saving trained model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "base_model": base_model,
        "output_dir": str(output_dir),
        "train_examples": len(train_tokenized),
        "valid_examples": len(valid_tokenized) if has_validation else 0,
        "max_target_length": max_target_length,
        "gpu_available": torch.cuda.is_available(),
        "directions": ["en2ne", "ne2en"],
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune NLLB for bidirectional EN↔NE translation."
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
    args = parser.parse_args()
    set_seed(args.seed)

    config = load_config(Path(args.config))
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    print(f"Project root: {PROJECT_ROOT}")

    data_dir = resolve_project_path(
        args.data_dir or data_cfg.get("data_dir"),
        DEFAULT_DATA_DIR,
    )
    output_dir = resolve_project_path(
        args.output_dir or training_cfg.get("output_dir"),
        DEFAULT_OUTPUT_DIR,
    )
    base_model = args.base_model or training_cfg.get("base_model", DEFAULT_BASE_MODEL)
    max_source_length = int(training_cfg.get("max_source_length", 160))
    max_target_length = int(training_cfg.get("max_target_length", 160))
    validation_split = float(training_cfg.get("validation_split", 0.1))

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    examples = load_examples(data_dir)
    examples = deduplicate_examples(examples)
    if not examples:
        raise RuntimeError(f"No usable EN/NE translation pairs found in {data_dir}")

    random.Random(args.seed).shuffle(examples)
    output_dir.mkdir(parents=True, exist_ok=True)

    en2ne_count = sum(1 for e in examples if e.get("direction") == "en2ne")
    ne2en_count = sum(1 for e in examples if e.get("direction") == "ne2en")
    print(
        f"Loaded {len(examples)} usable pairs from {data_dir}  "
        f"(EN→NE: {en2ne_count}, NE→EN: {ne2en_count})"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    train_tokenized, valid_tokenized, valid_raw = build_datasets(
        examples,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        validation_split=validation_split,
        seed=args.seed,
    )

    prepared_summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "example_count": len(examples),
        "en2ne_count": en2ne_count,
        "ne2en_count": ne2en_count,
        "file_count": len(list(iter_tabular_files(data_dir))),
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(prepared_summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(prepared_summary, ensure_ascii=False, indent=2))

    if args.prepare_only:
        print("Preparation complete. Skipping training (--prepare-only).")
        return

    summary = train_model(
        train_tokenized, valid_tokenized, valid_raw, config, output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    main()
