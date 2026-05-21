# NP <-> EN Translation Pipeline

This project trains and evaluates a Nepali-English translation model based on NLLB.

## Training Process

1. Collect bilingual TSV files from the `Training/data/` directory.
2. Normalize text, remove duplicates, and build a prepared corpus.
3. Split the data into training and validation sets.
4. Tokenize source and target text for NLLB.
5. Fine-tune `facebook/nllb-200-distilled-600M` with `Seq2SeqTrainer`.
6. Save checkpoints and the final model to `Training/checkpoints/translation_model/`.
7. Evaluate the checkpoint with BLEU, chrF, and strict matching metrics.
8. Upload the trained model folder to Hugging Face.

## Key Files

- Training script: [Training/scripts/train_translation.py](Training/scripts/train_translation.py)
- Evaluation script: [Training/scripts/evaluate_translation.py](Training/scripts/evaluate_translation.py)
- Training config: [Training/configs/training_config.yaml](Training/configs/training_config.yaml)
- Upload script: [uploadtoHF](uploadtoHF)

## Hugging Face Model

The model repository is here:

- [SarjakBhandari-230383/EchoNep](https://huggingface.co/SarjakBhandari-230383/EchoNep)

Important uploaded files include:

- [model.safetensors](https://huggingface.co/SarjakBhandari-230383/EchoNep/blob/main/model.safetensors)
- [config.json](https://huggingface.co/SarjakBhandari-230383/EchoNep/blob/main/config.json)
- [tokenizer.json](https://huggingface.co/SarjakBhandari-230383/EchoNep/blob/main/tokenizer.json)

