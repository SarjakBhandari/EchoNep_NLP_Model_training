#!/usr/bin/env python3
"""Generate a simple Markdown report summarizing dataset splits and evaluation metrics."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import yaml
import statistics
import json as _json
from collections import Counter
from typing import List


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open('r', encoding='utf-8') as fh:
        return sum(1 for _ in fh) - 1  # subtract header line


def load_training_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def load_metrics(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data'
    processed = data_dir / 'processed'
    reports_dir = root / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_file = processed / 'train.jsonl'
    valid_file = processed / 'valid.jsonl'
    test_file = processed / 'test.jsonl'
    tsv_test = data_dir / 'tests.tsv'

    train_count = count_lines(train_file)
    valid_count = count_lines(valid_file)
    test_count = count_lines(test_file) or count_lines(tsv_test)

    config = load_training_config(root / 'configs' / 'training_config.yaml')
    metrics = load_metrics(reports_dir / 'model_eval.json')

    # prefer human-run metrics if present
    human_metrics_path = reports_dir / 'model_eval_human.json'
    if human_metrics_path.exists():
        metrics = load_metrics(human_metrics_path)

    md_lines = [
        '# Translation Evaluation Report',
        '',
        '## Dataset splits',
        '',
        f'- Training examples: **{train_count}**',
        f'- Validation examples: **{valid_count}**',
        f'- Test examples: **{test_count}**',
        '',
        '## Training config (selected)',
        '',
    ]

    for k in ['output_dir', 'base_model', 'max_source_length', 'max_target_length', 'batch_size', 'num_train_epochs', 'learning_rate']:
        val = config.get('training', {}).get(k)
        if val is not None:
            md_lines.append(f'- **{k}**: {val}')

    md_lines += ['', '## Evaluation metrics', '']
    if metrics:
        md_lines.append(f"- Total examples evaluated: **{metrics.get('total_examples', 'N/A')}**")
        bleu = metrics.get('BLEU') or metrics.get('BLEU_score')
        if isinstance(bleu, dict):
            md_lines.append(f"- BLEU: **{bleu.get('score', 'N/A'):.2f}**")
        else:
            try:
                md_lines.append(f"- BLEU: **{float(bleu):.2f}**")
            except Exception:
                pass
        chrf = metrics.get('chrF', {}).get('score')
        if chrf is not None:
            md_lines.append(f"- chrF: **{chrf:.2f}**")
        ter = metrics.get('TER', {}).get('score')
        if ter is not None:
            md_lines.append(f"- TER: **{ter:.2f}**")
        md_lines.append('')
    else:
        md_lines.append('- No metrics found at `reports/model_eval.json`.')

    md_lines += ['## Notes', '', '- Evaluation used sacrebleu. BLEU is reference-dependent.', "- Ensure test references are human-authored for unbiased thesis results."]

    # Add sample-level mismatches table if predictions exist
    preds_file = reports_dir / 'predictions.jsonl'
    mismatches: List[Dict] = []
    if preds_file.exists():
        try:
            with preds_file.open('r', encoding='utf-8') as fh:
                for idx, ln in enumerate(fh):
                    if not ln.strip():
                        continue
                    j = _json.loads(ln)
                    ref = str(j.get('reference','')).strip()
                    pred = str(j.get('prediction','')).strip()
                    if ref != pred:
                        mismatches.append({'index': idx, 'source': j.get('source',''), 'reference': ref, 'prediction': pred})
        except Exception:
            mismatches = []

    if mismatches:
        md_lines += ['', '## Sample-level mismatches (first 10)', '']
        md_lines.append('| # | Source | Reference | Prediction |')
        md_lines.append('|---|--------|-----------|------------|')
        for m in mismatches[:10]:
            src = m['source'].replace('|','\|') if m['source'] else ''
            ref = m['reference'].replace('|','\|')
            pred = m['prediction'].replace('|','\|')
            md_lines.append(f"| {m['index']} | {src} | {ref} | {pred} |")

    out = reports_dir / 'translation_report.md'
    # Attempt to plot figures (counts, metrics, length distributions)
    figs_dir = reports_dir / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)

    def try_plot():
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return False

        # 1) Train/Valid/Test counts
        labels = ['train', 'valid', 'test']
        values = [train_count, valid_count, test_count]
        plt.figure(figsize=(6,4))
        plt.bar(labels, values, color=['#4c72b0','#55a868','#c44e52'])
        plt.title('Dataset split counts')
        plt.ylabel('Examples')
        plt.tight_layout()
        counts_fig = figs_dir / 'train_test_counts.png'
        plt.savefig(counts_fig)
        plt.close()

        # 2) Metrics bar chart
        metric_vals = {}
        if metrics:
            # extract candidate metric numbers
            def get_score(k):
                v = metrics.get(k)
                if isinstance(v, dict):
                    return float(v.get('score', float('nan')))
                try:
                    return float(v)
                except Exception:
                    return float('nan')
            metric_vals = {'BLEU': get_score('BLEU') or get_score('BLEU_score'),
                           'chrF': get_score('chrF'), 'TER': get_score('TER')}
        if metric_vals:
            names = [k for k in metric_vals.keys() if not (metric_vals[k] is None or (isinstance(metric_vals[k], float) and (metric_vals[k] != metric_vals[k])))]
            vals = [metric_vals[n] for n in names]
            plt.figure(figsize=(6,4))
            plt.bar(names, vals, color=['#4c72b0','#55a868','#c44e52'])
            plt.title('Evaluation metrics')
            plt.tight_layout()
            metrics_fig = figs_dir / 'metrics.png'
            plt.savefig(metrics_fig)
            plt.close()

        # 3) Source length distributions (train vs test)
        def load_sources_from_jsonl(p: Path) -> List[str]:
            if not p.exists():
                return []
            out: List[str] = []
            try:
                with p.open('r', encoding='utf-8') as fh:
                    for ln in fh:
                        if not ln.strip():
                            continue
                        try:
                            j = _json.loads(ln)
                            if isinstance(j, dict):
                                s = j.get('source') or j.get('text') or list(j.values())[0]
                                out.append(str(s))
                        except Exception:
                            continue
            except Exception:
                return []
            return out

        train_srcs = load_sources_from_jsonl(train_file)
        test_srcs = load_sources_from_jsonl(test_file)
        if not train_srcs and tsv_test.exists():
            try:
                import pandas as _pd
                df = _pd.read_csv(tsv_test, sep='\t', header=0)
                test_srcs = df.iloc[:,0].astype(str).tolist()
            except Exception:
                pass

        if train_srcs or test_srcs:
            train_lens = [len(s.split()) for s in train_srcs]
            test_lens = [len(s.split()) for s in test_srcs]
            plt.figure(figsize=(6,4))
            if train_lens:
                plt.hist(train_lens, bins=30, alpha=0.6, label='train')
            if test_lens:
                plt.hist(test_lens, bins=30, alpha=0.6, label='test')
            plt.legend()
            plt.xlabel('Source token count')
            plt.ylabel('Frequency')
            plt.title('Source length distribution')
            plt.tight_layout()
            length_fig = figs_dir / 'length_distribution.png'
            plt.savefig(length_fig)
            plt.close()

        return True

    plotted = try_plot()
    # Embed figures if plotted
    if plotted:
        md_lines += ['', '## Figures', '']
        md_lines.append('![Train/Valid/Test counts](reports/figures/train_test_counts.png)')
        md_lines.append('')
        md_lines.append('![Evaluation metrics](reports/figures/metrics.png)')
        md_lines.append('')
        md_lines.append('![Source length distribution](reports/figures/length_distribution.png)')

    out.write_text('\n'.join(md_lines), encoding='utf-8')
    print('Wrote', out)


if __name__ == '__main__':
    main()
