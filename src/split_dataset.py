import random
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _clean_text(value) -> str:
    """Convert value to stripped string; return empty string when missing."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def aggregate_queries(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Aggregate data by query to compute statistics for stratification."""

    print("Aggregating data by query...")

    # Group by query and compute statistics
    query_stats = df.groupby(cfg.dataset.fields.query).agg({
        cfg.dataset.fields.language: lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],  # Language mode
        cfg.dataset.fields.impressions: 'sum',  # Total impressions
        cfg.dataset.fields.selects: 'sum',      # Total selects
        cfg.dataset.fields.query: 'size'        # Row count per query
    }).rename(columns={
        cfg.dataset.fields.language: 'language',
        cfg.dataset.fields.impressions: 'total_impressions',
        cfg.dataset.fields.selects: 'total_selects',
        cfg.dataset.fields.query: 'row_count'
    }).reset_index()

    query_stats['query'] = query_stats[cfg.dataset.fields.query]

    print(f"Found {len(query_stats):,} unique queries")

    # Assign frequency bands based on total selects
    query_stats = assign_frequency_bands(query_stats, cfg)

    return query_stats


def assign_frequency_bands(query_stats: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Assign frequency bands (Head/Mid/Tail) to queries."""

    print("Assigning frequency bands...")

    # Use total selects for frequency ranking
    freq_col = 'total_selects'

    # Calculate percentiles for banding
    head_threshold = query_stats[freq_col].quantile(1 - cfg.dataset.split.frequency_bands.head_ratio)
    mid_threshold = query_stats[freq_col].quantile(cfg.dataset.split.frequency_bands.tail_ratio)

    def get_band(selects: int) -> str:
        if selects >= head_threshold:
            return 'Head'
        elif selects >= mid_threshold:
            return 'Mid'
        else:
            return 'Tail'

    query_stats['frequency_band'] = query_stats[freq_col].apply(get_band)

    # Print band statistics
    band_counts = query_stats['frequency_band'].value_counts()
    band_stats = query_stats.groupby('frequency_band')[freq_col].agg(['count', 'min', 'max', 'mean'])

    print("\nFrequency Band Statistics:")
    for band in ['Head', 'Mid', 'Tail']:
        if band in band_stats.index:
            stats = band_stats.loc[band]
            print(f"  {band}: {stats['count']:,} queries, selects range [{stats['min']:.0f}, {stats['max']:.0f}], mean {stats['mean']:.1f}")

    return query_stats


def stratified_query_split(query_stats: pd.DataFrame, cfg: DictConfig) -> Tuple[List[str], List[str], List[str]]:
    """Split queries into train/val/test with stratification by language and frequency band."""

    print("Performing stratified query-level split...")

    # Create stratification labels
    query_stats['stratify_label'] = query_stats['language'] + '_' + query_stats['frequency_band']

    # Get unique queries and their stratification labels
    queries = query_stats['query'].tolist()
    stratify_labels = query_stats['stratify_label'].tolist()

    print(f"Stratification groups: {len(set(stratify_labels))} unique combinations")

    # First split: train+val vs test
    train_val_queries, test_queries = train_test_split(
        queries,
        test_size=cfg.dataset.split.test_ratio,
        random_state=cfg.seed,
        stratify=stratify_labels
    )

    # Get stratification labels for train+val queries
    train_val_mask = query_stats['query'].isin(train_val_queries)
    train_val_labels = query_stats[train_val_mask]['stratify_label'].tolist()

    # Second split: train vs val
    val_ratio_adjusted = cfg.dataset.split.val_ratio / (1 - cfg.dataset.split.test_ratio)
    train_queries, val_queries = train_test_split(
        train_val_queries,
        test_size=val_ratio_adjusted,
        random_state=cfg.seed,
        stratify=train_val_labels
    )

    print(f"Query split: Train {len(train_queries):,}, Val {len(val_queries):,}, Test {len(test_queries):,}")

    return train_queries, val_queries, test_queries


def create_row_splits(df: pd.DataFrame, train_queries: List[str], val_queries: List[str],
                     test_queries: List[str], cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create row-level splits based on query assignments."""

    query_col = cfg.dataset.fields.query

    train_df = df[df[query_col].isin(train_queries)].copy()
    val_df = df[df[query_col].isin(val_queries)].copy()
    test_df = df[df[query_col].isin(test_queries)].copy()

    # Verify no query leakage
    train_queries_set = set(train_df[query_col])
    val_queries_set = set(val_df[query_col])
    test_queries_set = set(test_df[query_col])

    assert len(train_queries_set & val_queries_set) == 0, "Query leakage between train and val!"
    assert len(train_queries_set & test_queries_set) == 0, "Query leakage between train and test!"
    assert len(val_queries_set & test_queries_set) == 0, "Query leakage between val and test!"

    return train_df, val_df, test_df


def split_dataset(cfg: DictConfig) -> None:
    """Split dataset into train/val/test sets by query with stratification."""

    print("Loading dataset...")
    path = to_absolute_path(cfg.dataset.path)
    df = pd.read_parquet(path)

    print(f"Original dataset size: {len(df):,} rows")

    # Apply filtering
    df = df[
        (df[cfg.dataset.fields.selects] >= cfg.dataset.sampling.min_selects)
        & (df[cfg.dataset.fields.impressions] >= cfg.dataset.sampling.min_impressions)
    ]

    if cfg.dataset.sampling.max_rows is not None:
        # Ensure we don't sample more than available
        sample_size = min(cfg.dataset.sampling.max_rows, len(df))
        if sample_size < cfg.dataset.sampling.max_rows:
            print(f"Warning: Requested {cfg.dataset.sampling.max_rows:,} samples but only {len(df):,} available. Using all available data.")
        df = df.sample(n=sample_size, random_state=cfg.seed)
    else:
        print("Using all available data (no sampling limit)")

    print(f"Filtered dataset size: {len(df):,} rows")
    print(f"Unique queries: {df[cfg.dataset.fields.query].nunique():,}")

    # Aggregate by query for stratification
    query_stats = aggregate_queries(df, cfg)

    # Split queries with stratification
    train_queries, val_queries, test_queries = stratified_query_split(query_stats, cfg)

    # Create row-level splits
    train_df, val_df, test_df = create_row_splits(df, train_queries, val_queries, test_queries, cfg)

    print(f"\nFinal splits:")
    print(f"Train set: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%), {len(train_queries):,} queries")
    print(f"Val set: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%), {len(val_queries):,} queries")
    print(f"Test set: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%), {len(test_queries):,} queries")

    # Create output directory
    output_dir = Path(to_absolute_path(cfg.dataset.split.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved splits to:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")

    # Print detailed statistics
    print("\nDetailed Statistics:")
    for name, split_df, queries in [("Train", train_df, train_queries), ("Val", val_df, val_queries), ("Test", test_df, test_queries)]:
        print(f"\n{name} Set:")
        print(f"  Rows: {len(split_df):,}")
        print(f"  Queries: {len(queries):,}")

        # Language distribution
        lang_counts = split_df[cfg.dataset.fields.language].value_counts()
        print(f"  Language distribution: {lang_counts.to_dict()}")

        # Query frequency band distribution
        query_freq_bands = query_stats[query_stats['query'].isin(queries)]['frequency_band'].value_counts()
        print(f"  Frequency bands: {query_freq_bands.to_dict()}")

        # Query statistics
        query_rows = split_df.groupby(cfg.dataset.fields.query).size()
        print(f"  Avg rows per query: {query_rows.mean():.1f}")
        print(f"  Query row count distribution: min={query_rows.min()}, max={query_rows.max()}, median={query_rows.median():.0f}")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    split_dataset(cfg)


if __name__ == "__main__":
    main()
