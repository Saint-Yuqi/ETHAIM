#!/usr/bin/env python3
"""
Simple verification script to check query-level split integrity.
Run this after split_dataset.py to verify no query leakage.
"""

import pandas as pd
from pathlib import Path

def verify_query_splits():
    """Verify that no queries appear in multiple splits."""

    data_dir = Path("./data/splits")

    if not data_dir.exists():
        print("Error: ./data/splits directory not found. Run split_dataset.py first.")
        return False

    # Load splits
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    # Get unique queries from each split
    train_queries = set(train_df['OriginalQuery'].unique())
    val_queries = set(val_df['OriginalQuery'].unique())
    test_queries = set(test_df['OriginalQuery'].unique())

    print(f"Train queries: {len(train_queries):,}")
    print(f"Val queries: {len(val_queries):,}")
    print(f"Test queries: {len(test_queries):,}")

    # Check for overlaps
    train_val_overlap = train_queries & val_queries
    train_test_overlap = train_queries & test_queries
    val_test_overlap = val_queries & test_queries

    all_overlaps = train_val_overlap | train_test_overlap | val_test_overlap

    if all_overlaps:
        print("❌ ERROR: Query leakage detected!")
        if train_val_overlap:
            print(f"  Train-Val overlap: {len(train_val_overlap)} queries")
        if train_test_overlap:
            print(f"  Train-Test overlap: {len(train_test_overlap)} queries")
        if val_test_overlap:
            print(f"  Val-Test overlap: {len(val_test_overlap)} queries")
        return False
    else:
        print("✅ SUCCESS: No query leakage detected!")

    # Print statistics
    total_queries = len(train_queries) + len(val_queries) + len(test_queries)
    total_rows = len(train_df) + len(val_df) + len(test_df)

    print("Dataset Statistics:")
    print(f"  Total unique queries: {total_queries:,}")
    print(f"  Total rows: {total_rows:,}")

    return True

if __name__ == "__main__":
    verify_query_splits()
