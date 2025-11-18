import random
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, util


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_product_text(row: pd.Series, cfg) -> str:
    product_name = _clean_text(row[cfg.dataset.fields.product_name])
    brand = _clean_text(row[cfg.dataset.fields.brand])
    level4 = _clean_text(row[cfg.dataset.fields.level4_group])
    product_type = _clean_text(row[cfg.dataset.fields.product_type])

    parts = [product_name]
    if brand:
        parts.append(brand)

    tail = level4 if level4 else product_type
    if tail:
        parts.append(tail)

    return " | ".join(parts)


def evaluate_model(model: SentenceTransformer, cfg, eval_df: pd.DataFrame) -> Tuple[float, float]:
    ranks = []
    hits = []

    for _, group in eval_df.groupby(cfg.dataset.fields.query):
        if group.empty:
            continue

        if len(group) > cfg.eval.max_products_per_query:
            group = group.sample(
                n=cfg.eval.max_products_per_query, random_state=cfg.seed
            )

        best_index = group[cfg.dataset.fields.selects].idxmax()

        queries = group[cfg.dataset.fields.query].iloc[0]
        q_text = f"{cfg.model.use_prefix.query}{queries}"

        product_rows = list(group.iterrows())
        product_texts = []
        best_row_idx = None
        for i, (idx, row) in enumerate(product_rows):
            product_body = build_product_text(row, cfg)
            product_texts.append(f"{cfg.model.use_prefix.product}{product_body}")
            if idx == best_index:
                best_row_idx = i

        if best_row_idx is None:
            continue

        q_vec = model.encode(
            [q_text],
            device=cfg.train.device,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        p_vecs = model.encode(
            product_texts,
            device=cfg.train.device,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        scores = util.cos_sim(q_vec, p_vecs).squeeze(0)
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == best_row_idx).nonzero(as_tuple=False).item() + 1

        ranks.append(rank)
        hits.append(1 if rank <= cfg.eval.metrics.hit_k else 0)

    avg_rank = float(sum(ranks) / len(ranks)) if ranks else 0.0
    hit_k = float(sum(hits) / len(hits)) if hits else 0.0

    return avg_rank, hit_k


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    path = to_absolute_path(cfg.dataset.path)
    df = pd.read_parquet(path)
    df = df[df[cfg.dataset.fields.selects] >= cfg.dataset.sampling.min_selects]

    unique_queries = df[cfg.dataset.fields.query].unique()
    query_series = pd.Series(unique_queries)
    if len(query_series) > cfg.eval.max_queries:
        selected_queries = query_series.sample(
            n=cfg.eval.max_queries, random_state=cfg.seed
        )
    else:
        selected_queries = query_series

    eval_df = df[df[cfg.dataset.fields.query].isin(selected_queries)]
    print(f"Eval queries: {eval_df[cfg.dataset.fields.query].nunique()}")

    base_model = SentenceTransformer(cfg.model.base_model_name)
    if cfg.train.device and cfg.train.device != "cpu":
        base_model = base_model.to(cfg.train.device)
    avg_rank_base, hit_base = evaluate_model(base_model, cfg, eval_df)
    print(f"[Base] avg_rank_best = {avg_rank_base:.4f}, Hit@K = {hit_base:.4f}")

    finetuned_path = to_absolute_path(cfg.train.output.dir)
    ft_model = SentenceTransformer(finetuned_path)
    if cfg.train.device and cfg.train.device != "cpu":
        ft_model = ft_model.to(cfg.train.device)
    avg_rank_ft, hit_ft = evaluate_model(ft_model, cfg, eval_df)
    print(f"[Finetuned] avg_rank_best = {avg_rank_ft:.4f}, Hit@K = {hit_ft:.4f}")


if __name__ == "__main__":
    main()
