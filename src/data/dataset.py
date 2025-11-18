from typing import List, Tuple

import pandas as pd
from hydra.utils import to_absolute_path
from sentence_transformers import InputExample
from torch.utils.data import Dataset


def _clean_text(value) -> str:
    """Convert value to stripped string; return empty string when missing."""
    if pd.isna(value):
        return ""
    return str(value).strip()


class EmbeddingPairsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg_dataset, cfg_model) -> None:
        self.df = df.reset_index(drop=True)
        self.cfg_dataset = cfg_dataset
        self.cfg_model = cfg_model

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> InputExample:
        row = self.df.iloc[idx]
        query = _clean_text(row[self.cfg_dataset.fields.query])
        product_text = self._build_product_text(row)

        query_text = f"{self.cfg_model.use_prefix.query}{query}"
        product_text = f"{self.cfg_model.use_prefix.product}{product_text}"

        return InputExample(texts=[query_text, product_text])

    def _build_product_text(self, row: pd.Series) -> str:
        product_name = _clean_text(row[self.cfg_dataset.fields.product_name])
        brand = _clean_text(row[self.cfg_dataset.fields.brand])
        level4 = _clean_text(row[self.cfg_dataset.fields.level4_group])
        product_type = _clean_text(row[self.cfg_dataset.fields.product_type])

        parts = [product_name]
        if brand:
            parts.append(brand)

        tail = level4 if level4 else product_type
        if tail:
            parts.append(tail)

        return " | ".join(parts)


def load_train_eval_examples(cfg) -> Tuple[List[InputExample], List[InputExample]]:
    """Load train and validation examples from predefined splits."""

    if cfg.dataset.split.method == "predefined":
        # Load from predefined splits
        train_path = to_absolute_path(f"{cfg.dataset.split.output_dir}/train.parquet")
        val_path = to_absolute_path(f"{cfg.dataset.split.output_dir}/val.parquet")

        train_df = pd.read_parquet(train_path)
        eval_df = pd.read_parquet(val_path)

        print(f"Loaded predefined splits:")
        print(f"  Train: {len(train_df):,} examples")
        print(f"  Val: {len(eval_df):,} examples")

    else:
        # Legacy random split (for backward compatibility)
        print("Warning: Using random split. Consider using predefined splits for reproducibility.")

        path = to_absolute_path(cfg.dataset.path)
        df = pd.read_parquet(path)

        df = df[
            (df[cfg.dataset.fields.selects] >= cfg.dataset.sampling.min_selects)
            & (df[cfg.dataset.fields.impressions] >= cfg.dataset.sampling.min_impressions)
        ]

        if cfg.dataset.sampling.max_rows is not None:
            df = df.sample(n=cfg.dataset.sampling.max_rows, random_state=cfg.seed)

        eval_df = df.sample(frac=cfg.dataset.split.eval_ratio, random_state=cfg.seed)
        train_df = df.drop(eval_df.index)

    train_dataset = EmbeddingPairsDataset(train_df, cfg.dataset, cfg.model)
    eval_dataset = EmbeddingPairsDataset(eval_df, cfg.dataset, cfg.model)

    train_examples = [train_dataset[i] for i in range(len(train_dataset))]
    eval_examples = [eval_dataset[i] for i in range(len(eval_dataset))]

    return train_examples, eval_examples


def load_test_examples(cfg) -> List[InputExample]:
    """Load test examples from predefined test split."""

    if cfg.dataset.split.method != "predefined":
        raise ValueError("Test split only available when using predefined splits")

    test_path = to_absolute_path(f"{cfg.dataset.split.output_dir}/test.parquet")
    test_df = pd.read_parquet(test_path)

    print(f"Loaded test split: {len(test_df):,} examples")

    test_dataset = EmbeddingPairsDataset(test_df, cfg.dataset, cfg.model)
    test_examples = [test_dataset[i] for i in range(len(test_dataset))]

    return test_examples
