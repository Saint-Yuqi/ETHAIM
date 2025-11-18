import json
import random
from typing import Dict, List

import hydra
import numpy as np
import pandas as pd
import requests
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


def build_prompt(query: str, results_a: List[str], results_b: List[str]) -> str:
    lines = [
        "You are evaluating search results for an e-commerce website.",
        "",
        "User query:",
        f"\"{query}\"",
        "",
        "System A results (ranked from 1 to {k}):".format(k=len(results_a)),
    ]
    for i, text in enumerate(results_a, start=1):
        lines.append(f"{i}. {text}")

    lines.append("")
    lines.append("System B results (ranked from 1 to {k}):".format(k=len(results_b)))
    for i, text in enumerate(results_b, start=1):
        lines.append(f"{i}. {text}")

    lines.extend(
        [
            "",
            "Please answer with a single JSON object with the following fields:",
            '- "winner": "A", "B", or "tie"',
            '- "reason": a short explanation in 1-2 sentences',
            "",
            "Judge which system provides more relevant and appropriate results for the given query.",
            "Consider both relevance and avoiding obviously wrong or inappropriate categories.",
        ]
    )

    return "\n".join(lines)


def call_llm(prompt: str, cfg) -> Dict[str, str]:
    payload = {
        "model": cfg.eval_llm.llm.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.eval_llm.llm.temperature,
        "max_tokens": cfg.eval_llm.llm.max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {cfg.eval_llm.llm.api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(cfg.eval_llm.llm.api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON object if wrapped
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


def get_topk_indices(
    model: SentenceTransformer,
    query_text: str,
    product_texts: List[str],
    device: str,
    k: int,
) -> List[int]:
    if not product_texts:
        return []
    top_k = min(k, len(product_texts))
    q_vec = model.encode(query_text, convert_to_tensor=True, device=device)
    p_vecs = model.encode(product_texts, convert_to_tensor=True, device=device)
    scores = util.cos_sim(q_vec, p_vecs)[0].cpu().numpy()
    order = scores.argsort()[::-1][:top_k]
    return order.tolist()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    if not cfg.eval_llm.llm.api_key:
        raise ValueError("NUWA_API_KEY is missing; please set it in env or config.")
    if not cfg.eval_llm.llm.model:
        raise ValueError("LLM model name is missing; please set eval_llm.llm.model.")

    test_path = to_absolute_path(cfg.paths.test_split)
    df = pd.read_parquet(test_path)
    df = df[df[cfg.dataset.fields.selects] >= cfg.dataset.sampling.min_selects]

    unique_queries = df[cfg.dataset.fields.query].unique()
    if len(unique_queries) > cfg.eval_llm.num_queries:
        sampled_queries = np.random.choice(
            unique_queries, size=cfg.eval_llm.num_queries, replace=False
        )
    else:
        sampled_queries = unique_queries

    base_model = SentenceTransformer(cfg.model.base_model_name)
    if cfg.train.device and cfg.train.device != "cpu":
        base_model = base_model.to(cfg.train.device)

    ft_model = SentenceTransformer(to_absolute_path(cfg.train.output.dir))
    if cfg.train.device and cfg.train.device != "cpu":
        ft_model = ft_model.to(cfg.train.device)

    winners = []
    reasons = []
    for q in sampled_queries:
        group = df[df[cfg.dataset.fields.query] == q]
        if group.empty:
            continue

        product_texts = [
            build_product_text(row, cfg) for _, row in group.iterrows()
        ]
        prefixed_products = [f"{cfg.model.use_prefix.product}{t}" for t in product_texts]
        q_text = f"{cfg.model.use_prefix.query}{q}"

        order_a = get_topk_indices(
            base_model, q_text, prefixed_products, cfg.train.device, cfg.eval_llm.top_k
        )
        order_b = get_topk_indices(
            ft_model, q_text, prefixed_products, cfg.train.device, cfg.eval_llm.top_k
        )

        results_a = [product_texts[i] for i in order_a]
        results_b = [product_texts[i] for i in order_b]

        prompt = build_prompt(q, results_a, results_b)
        llm_response = call_llm(prompt, cfg)
        winner = llm_response.get("winner", "").lower()
        reason = llm_response.get("reason", "")
        winners.append(winner)
        reasons.append(reason)

    total = len(winners)
    num_a = sum(1 for w in winners if w == "a")
    num_b = sum(1 for w in winners if w == "b")
    num_tie = sum(1 for w in winners if w == "tie")

    if total == 0:
        print("No queries evaluated.")
        return

    print(f"LLM prefers baseline (A): {num_a/total:.3f}")
    print(f"LLM prefers finetuned (B): {num_b/total:.3f}")
    print(f"LLM says tie: {num_tie/total:.3f}")


if __name__ == "__main__":
    main()
