"""
Validate that pruned subsets preserve go/no-go signal vs full benchmark.
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from evalscope.pruners.base_pruner import extract_score
from evalscope.pruners.lcb_pruner import LiveCodeBenchPruner
from evalscope.pruners.aalcr_pruner import AALCRPruner


def compute_model_scores(reviews_dir: str, benchmark_prefix: str, pruned_indices=None) -> dict:
    """
    Compute per-model full and pruned aggregate scores.
    Only loads files matching benchmark_prefix to avoid cross-benchmark contamination.
    """
    reviews_dir = Path(reviews_dir)
    pattern     = f'{benchmark_prefix}__*.jsonl'
    files       = sorted(reviews_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f'No files matching {pattern} in {reviews_dir}')

    pruned_set  = set(pruned_indices) if pruned_indices else None
    model_scores = {}

    for fpath in files:
        stem       = fpath.stem                                    # e.g. live_code_bench_v5__gpt-oss-120b
        model_name = stem.split('__', 1)[-1] if '__' in stem else stem

        all_scores    = []
        pruned_scores = []

        with open(fpath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                idx   = row.get('index')
                score = extract_score(row)
                if idx is None or score is None:
                    continue

                all_scores.append(score)
                if pruned_set is None or idx in pruned_set:
                    pruned_scores.append(score)

        if all_scores:
            model_scores[model_name] = {
                'full':     np.mean(all_scores),
                'pruned':   np.mean(pruned_scores) if pruned_scores else 0.0,
                'n_full':   len(all_scores),
                'n_pruned': len(pruned_scores),
            }

    return model_scores


def print_report(title: str, pruner, reviews_dir: str, benchmark_prefix: str, threshold: float):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')

    model_scores = compute_model_scores(reviews_dir, benchmark_prefix, pruner.pruned_indices)

    if not model_scores:
        print('  No model scores found.')
        return

    full_scores   = [v['full']   for v in model_scores.values()]
    pruned_scores = [v['pruned'] for v in model_scores.values()]

    print(f'\n  {"Model":<25} {"Full":>8} {"Pruned":>8} {"Delta":>8}  {"GO/NO-GO"}')
    print(f'  {"-"*65}')

    agreements = []
    for model, s in sorted(model_scores.items()):
        f, p   = s['full'], s['pruned']
        delta  = p - f
        fg     = 'GO'    if f >= threshold else 'NO-GO'
        pg     = 'GO'    if p >= threshold else 'NO-GO'
        match  = '✓'     if fg == pg      else '✗ MISMATCH'
        agreements.append(fg == pg)
        print(f'  {model:<25} {f:>8.3f} {p:>8.3f} {delta:>+8.3f}  {fg} → {pg} {match}')

    rho, pval = spearmanr(full_scores, pruned_scores) if len(full_scores) > 2 else (1.0, 0.0)
    n_total   = list(model_scores.values())[0]['n_full']
    n_pruned  = list(model_scores.values())[0]['n_pruned']

    print(f'\n  Samples kept:       {n_pruned} / {n_total}  ({n_pruned/n_total:.1%})')
    print(f'  Rank correlation:   ρ = {rho:.3f}  (p={pval:.3f})')
    print(f'  Go/No-Go agreement: {sum(agreements)/len(agreements):.1%}')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lcb-reviews',   required=True)
    parser.add_argument('--aalcr-reviews', required=True)
    parser.add_argument('--threshold',     type=float, default=0.5)
    args = parser.parse_args()

    lcb_pruner   = LiveCodeBenchPruner(reviews_dir=args.lcb_reviews)
    aalcr_pruner = AALCRPruner(reviews_dir=args.aalcr_reviews)

    print_report(
        'LiveCodeBench v5  (315 → ~50)',
        lcb_pruner, args.lcb_reviews,
        benchmark_prefix='live_code_bench_v5',
        threshold=args.threshold
    )
    print_report(
        'AA-LCR  (100 → ~25)',
        aalcr_pruner, args.aalcr_reviews,
        benchmark_prefix='aa_lcr',
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()