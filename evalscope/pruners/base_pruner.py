import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def extract_score(row: dict) -> Optional[float]:
    """
    Extract score from nested review format:
    row['sample_score']['score']['value'][main_score_name]
    main_score_name is 'acc' (AA-LCR) or 'pass' (LCB).
    """
    try:
        value = row['sample_score']['score']['value']
        key   = row['sample_score']['score'].get('main_score_name', 'acc')
        score = value.get(key, value.get('acc', value.get('pass')))
        return float(score) if score is not None else None
    except (KeyError, TypeError):
        return None


def extract_metadata(row: dict) -> dict:
    try:
        meta = row['sample_score']['sample_metadata']
        urls = meta.get('data_source_urls', '')
        return {
            'input_tokens': int(meta.get('input_tokens', 0)),
            'n_sources':    len(urls.split(';')) if urls else 1,
        }
    except (KeyError, TypeError):
        return {'input_tokens': 0, 'n_sources': 1}


def load_reviews(
    reviews_dir: str,
    benchmark_prefix: str = '',
) -> Tuple[dict, dict]:
    """
    Load review JSONL files matching benchmark_prefix.

    Filename convention: {benchmark}__{model}.jsonl
    Model name is extracted as everything after the first '__'.

    Args:
        reviews_dir:       directory containing *.jsonl review files
        benchmark_prefix:  e.g. 'aa_lcr' or 'live_code_bench_v5'
                           if empty, loads all *.jsonl files

    Returns:
        scores:   {index -> {model_name -> score}}
        metadata: {index -> {input_tokens, n_sources}}
    """
    reviews_dir = Path(reviews_dir)
    pattern     = f'{benchmark_prefix}*.jsonl' if benchmark_prefix else '*.jsonl'
    files       = sorted(reviews_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f'No files matching "{pattern}" in {reviews_dir}. '
            f'Files present: {[f.name for f in reviews_dir.glob("*.jsonl")]}'
        )

    scores   = {}
    metadata = {}

    for fpath in files:
        stem = fpath.stem  # e.g. 'aa_lcr__gpt-oss-120b'
        # Extract model name: everything after the first '__'
        model_name = stem.split('__', 1)[-1] if '__' in stem else stem

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

                if idx not in scores:
                    scores[idx]   = {}
                    metadata[idx] = extract_metadata(row)

                scores[idx][model_name] = score

    models = {m for v in scores.values() for m in v}
    print(f'[Pruner] Loaded {len(scores)} samples | '
          f'{len(files)} files | models: {sorted(models)}')
    return scores, metadata


def compute_sample_stats(scores: dict, metadata: dict) -> dict:
    stats = {}
    for idx, model_scores in scores.items():
        values = list(model_scores.values())
        if not values:
            continue
        stats[idx] = {
            'difficulty':     float(np.mean(values)),
            'discrimination': float(np.std(values)) if len(values) > 1 else 0.0,
            'n_models':       len(values),
            'input_tokens':   metadata.get(idx, {}).get('input_tokens', 0),
            'n_sources':      metadata.get(idx, {}).get('n_sources', 1),
        }
    return stats


def stratified_discrimination_sample(
    stats: dict,
    prune_ratio: float = 0.16,
    n_bins: int = 5,
    min_samples: int = 15,
    random_seed: int = 42,
) -> list:
    rng = np.random.default_rng(random_seed)

    indices      = list(stats.keys())
    difficulties = np.array([stats[i]['difficulty']     for i in indices])
    discs        = np.array([stats[i]['discrimination'] for i in indices])

    target_n = max(min_samples, int(len(indices) * prune_ratio))
    target_n = min(target_n, len(indices))

    bin_edges           = np.percentile(difficulties, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1]      += 1e-9
    bin_assignments     = np.digitize(difficulties, bin_edges[1:])

    selected = []

    for bin_id in range(n_bins):
        mask        = bin_assignments == bin_id
        bin_idx     = [indices[i] for i in range(len(indices)) if mask[i]]
        bin_discs   = [discs[i]   for i in range(len(indices)) if mask[i]]

        if not bin_idx:
            continue

        bin_target = max(1, round(target_n * len(bin_idx) / len(indices)))
        bin_target = min(bin_target, len(bin_idx))

        pairs    = sorted(zip(bin_idx, bin_discs), key=lambda x: x[1], reverse=True)
        n_disc   = max(1, int(bin_target * 0.70))
        n_rand   = bin_target - n_disc

        top      = [i for i, _ in pairs[:n_disc]]
        rest     = [i for i, _ in pairs[n_disc:]]
        rand     = list(rng.choice(rest, size=min(n_rand, len(rest)), replace=False)) if rest else []

        selected.extend(top + rand)

    final = sorted(set(selected))
    print(f'[Pruner] {len(final)} / {len(indices)} samples kept '
          f'({len(final)/len(indices):.1%})')
    return final