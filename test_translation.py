import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = r"Training/checkpoints/translation_model/checkpoint-3096"

print("=" * 80)
print("LOADING MODEL")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=False
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR
)

model.eval()

print("\nModel loaded from:")
print(MODEL_DIR)

print("\nModel config name:")
print(model.config._name_or_path)

print("\nFiles in model directory:")
for f in os.listdir(MODEL_DIR):
    print(" -", f)

print("\nGeneration Config:")
print("forced_bos_token_id =", model.generation_config.forced_bos_token_id)

print("\nLanguage Token IDs:")
print("eng_Latn =", tokenizer.convert_tokens_to_ids("eng_Latn"))
print("npi_Deva =", tokenizer.convert_tokens_to_ids("npi_Deva"))
print("unk_token =", tokenizer.unk_token_id)

print("\n" + "=" * 80)
print("TEST 1: ENGLISH -> NEPALI")
print("=" * 80)

text = "The price of this book is too expensive for me."

tokenizer.src_lang = "eng_Latn"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("npi_Deva"),
        max_new_tokens=128,
        num_beams=4
    )

print("\nInput:")
print(text)

print("\nRaw Output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=False))

print("\nClean Output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n" + "=" * 80)
print("TEST 2: NEPALI -> ENGLISH")
print("=" * 80)

text = "तिमीलाई कस्तो?"

tokenizer.src_lang = "npi_Deva"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_new_tokens=128,
        num_beams=4
    )

print("\nInput:")
print(text)

print("\nRaw Output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=False))

print("\nClean Output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n" + "=" * 80)
print("MANUAL TOKEN CHECK")
print("=" * 80)

for lang in ["eng_Latn", "npi_Deva"]:
    token_id = tokenizer.convert_tokens_to_ids(lang)
    print(f"{lang} -> {token_id}")

    if token_id == tokenizer.unk_token_id:
        print(f"WARNING: {lang} is unknown to tokenizer!")

print("\nDone.")