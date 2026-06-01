import numpy as np
from .base_pruner import load_reviews, compute_sample_stats, stratified_discrimination_sample


class AALCRPruner:
    """
    Stratified discrimination pruner for AA-LCR (100 samples → ~25).

    Loads files matching: aa_lcr__*.jsonl

    Extra design choices vs LCB:
    - Higher prune_ratio (0.25) to absorb LLM judge non-determinism
    - Context-length stratification: preserves short AND long context samples
      because AA-LCR specifically tests multi-document long-context reasoning
    """

    BENCHMARK_PREFIX   = 'aa_lcr'
    DEFAULT_PRUNE_RATIO = 0.25
    DEFAULT_MIN_SAMPLES = 20

    def __init__(
        self,
        reviews_dir: str,
        prune_ratio: float      = DEFAULT_PRUNE_RATIO,
        min_samples: int        = DEFAULT_MIN_SAMPLES,
        n_bins: int             = 4,
        random_seed: int        = 42,
        context_stratify: bool  = True,
    ):
        self.reviews_dir      = reviews_dir
        self.prune_ratio      = prune_ratio
        self.min_samples      = min_samples
        self.n_bins           = n_bins
        self.random_seed      = random_seed
        self.context_stratify = context_stratify
        self._pruned_indices  = None

    def _ensure_pruned(self):
        if self._pruned_indices is not None:
            return
        scores, metadata = load_reviews(self.reviews_dir, self.BENCHMARK_PREFIX)
        stats            = compute_sample_stats(scores, metadata)

        if self.context_stratify:
            self._pruned_indices = self._context_stratified(stats)
        else:
            self._pruned_indices = stratified_discrimination_sample(
                stats,
                prune_ratio = self.prune_ratio,
                n_bins      = self.n_bins,
                min_samples = self.min_samples,
                random_seed = self.random_seed,
            )

    def _context_stratified(self, stats: dict) -> list:
        """
        Two-axis stratification: difficulty × context_length quartile.

        Ensures the pruned set represents both short (~94k token) and
        long (~114k token) multi-document questions, since context length
        is itself a first-class capability dimension in AA-LCR.
        """
        rng      = np.random.default_rng(self.random_seed)
        indices  = list(stats.keys())
        ctx      = np.array([stats[i]['input_tokens']   for i in indices])
        discs    = np.array([stats[i]['discrimination'] for i in indices])

        target_n  = max(self.min_samples, int(len(indices) * self.prune_ratio))
        ctx_median = np.median(ctx)
        selected  = []

        for mask_fn, label in [
            (lambda: ctx <= ctx_median, 'short_ctx'),
            (lambda: ctx >  ctx_median, 'long_ctx'),
        ]:
            mask      = mask_fn()
            seg_idx   = [indices[i] for i in range(len(indices)) if mask[i]]
            seg_discs = [discs[i]   for i in range(len(indices)) if mask[i]]

            if not seg_idx:
                continue

            seg_target = max(3, round(target_n * len(seg_idx) / len(indices)))
            seg_target = min(seg_target, len(seg_idx))

            pairs  = sorted(zip(seg_idx, seg_discs), key=lambda x: x[1], reverse=True)
            n_disc = max(1, int(seg_target * 0.70))
            n_rand = seg_target - n_disc

            top    = [i for i, _ in pairs[:n_disc]]
            rest   = [i for i, _ in pairs[n_disc:]]
            rand   = list(rng.choice(rest, size=min(n_rand, len(rest)), replace=False)) if rest else []

            selected.extend(top + rand)
            print(f'[AALCRPruner] {label}: {len(top)+len(rand)} / {len(seg_idx)} kept')

        final = sorted(set(selected))
        print(f'[AALCRPruner] Total: {len(final)} / {len(indices)} kept')
        return final

    @property
    def pruned_indices(self) -> list:
        self._ensure_pruned()
        return self._pruned_indices

    @property
    def n_kept(self) -> int:
        return len(self.pruned_indices)

    def filter_samples(self, samples: list) -> list:
        pruned_set = set(self.pruned_indices)
        out = [s for s in samples if s.get('index') in pruned_set]
        print(f'[AALCRPruner] {len(samples)} → {len(out)} samples')
        return out

    def summary(self) -> dict:
        return {
            'benchmark':          'aa_lcr',
            'total':              100,
            'kept':               self.n_kept,
            'prune_ratio':        self.prune_ratio,
            'strategy':           'stratified_discrimination + context_stratification',
            'context_stratify':   self.context_stratify,
        }