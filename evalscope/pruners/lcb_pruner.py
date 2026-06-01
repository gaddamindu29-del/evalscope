from .base_pruner import load_reviews, compute_sample_stats, stratified_discrimination_sample


class LiveCodeBenchPruner:
    """
    Stratified discrimination pruner for LiveCodeBench v5 (315 samples → ~50).

    Loads files matching: live_code_bench_v5__*.jsonl
    """

    BENCHMARK_PREFIX   = 'live_code_bench_v5'
    DEFAULT_PRUNE_RATIO = 0.16
    DEFAULT_MIN_SAMPLES = 30

    def __init__(
        self,
        reviews_dir: str,
        prune_ratio: float = DEFAULT_PRUNE_RATIO,
        min_samples: int   = DEFAULT_MIN_SAMPLES,
        n_bins: int        = 5,
        random_seed: int   = 42,
    ):
        self.reviews_dir  = reviews_dir
        self.prune_ratio  = prune_ratio
        self.min_samples  = min_samples
        self.n_bins       = n_bins
        self.random_seed  = random_seed
        self._pruned_indices = None

    def _ensure_pruned(self):
        if self._pruned_indices is not None:
            return
        scores, metadata     = load_reviews(self.reviews_dir, self.BENCHMARK_PREFIX)
        stats                = compute_sample_stats(scores, metadata)
        self._pruned_indices = stratified_discrimination_sample(
            stats,
            prune_ratio = self.prune_ratio,
            n_bins      = self.n_bins,
            min_samples = self.min_samples,
            random_seed = self.random_seed,
        )

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
        print(f'[LCBPruner] {len(samples)} → {len(out)} samples')
        return out

    def summary(self) -> dict:
        return {
            'benchmark':   'live_code_bench_v5',
            'total':       315,
            'kept':        self.n_kept,
            'prune_ratio': self.prune_ratio,
            'strategy':    'stratified_discrimination',
        }