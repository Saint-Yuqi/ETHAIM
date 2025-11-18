import random

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, losses, evaluation, util
from torch.utils.data import DataLoader

from src.data.dataset import load_train_eval_examples, load_test_examples


class BiencoderEvaluator:
    """Custom evaluator for biencoder retrieval evaluation during training."""

    def __init__(self, eval_examples, name="val", show_progress_bar=True):
        self.eval_examples = eval_examples
        self.name = name
        self.show_progress_bar = show_progress_bar

        # For training evaluation, we'll use a simplified approach:
        # Randomly select "correct" products since we don't have selection labels in InputExamples
        # In production, you'd want to pass the actual selection information
        self.eval_data = []
        for example in eval_examples:
            query_text, product_text = example.texts
            self.eval_data.append((query_text, product_text))

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        """Evaluate model and return metrics using simplified ranking."""

        model.eval()

        # For training evaluation, we'll do a simplified assessment
        # Sample a few queries and evaluate their ranking performance
        sample_size = min(100, len(self.eval_data))  # Evaluate on sample for speed

        if sample_size < 10:
            # Not enough data for meaningful evaluation
            return {f"{self.name}_mrr": 0.0, f"{self.name}_hit@10": 0.0}

        # Sample evaluation data
        sampled_data = random.sample(self.eval_data, min(sample_size, len(self.eval_data)))

        ranks = []
        hits_at_10 = []

        with torch.no_grad():
            for query_text, correct_product_text in sampled_data:
                # Get other products as negatives (simplified approach)
                other_products = [p for q, p in self.eval_data if q == query_text and p != correct_product_text]
                if len(other_products) < 3:  # Need some negatives
                    continue

                # Sample a few negative products
                neg_products = np.random.choice(other_products, size=min(10, len(other_products)), replace=False)

                all_products = [correct_product_text] + list(neg_products)
                correct_idx = 0  # Correct product is always first

                # Encode
                query_vec = model.encode([query_text], convert_to_tensor=True, show_progress_bar=False)
                product_vecs = model.encode(all_products, convert_to_tensor=True, show_progress_bar=False)

                # Compute similarities
                scores = util.cos_sim(query_vec, product_vecs).squeeze(0)

                # Get ranking
                sorted_indices = torch.argsort(scores, descending=True)
                rank = (sorted_indices == correct_idx).nonzero(as_tuple=True)[0].item() + 1

                ranks.append(rank)
                hits_at_10.append(1 if rank <= 10 else 0)

        # Calculate metrics
        mrr = sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else 0.0
        hit_at_10 = sum(hits_at_10) / len(hits_at_10) if hits_at_10 else 0.0

        metrics = {
            f"{self.name}_mrr": mrr,
            f"{self.name}_hit@10": hit_at_10,
        }

        if self.show_progress_bar:
            print(f"  {self.name}_mrr: {mrr:.4f}, {self.name}_hit@10: {hit_at_10:.4f}")

        return metrics


def create_validation_evaluator(cfg, eval_examples):
    """Create BiencoderEvaluator for validation during training."""
    return BiencoderEvaluator(eval_examples, name="val", show_progress_bar=True)


def evaluate_on_test_set(cfg, model):
    """Evaluate trained model on test set using the same logic as eval_test.py."""

    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)

    try:
        # Import the evaluation function from eval_test.py
        from src.eval_test import evaluate_model_on_test

        # Load test data
        test_path = to_absolute_path(f"{cfg.dataset.split.output_dir}/test.parquet")
        test_df = pd.read_parquet(test_path)

        print(f"Test set: {len(test_df):,} examples, {test_df[cfg.dataset.fields.query].nunique()} unique queries")

        # Load base model for comparison
        base_model = SentenceTransformer(cfg.model.base_model_name)
        if cfg.train.device and cfg.train.device != "cpu":
            base_model = base_model.to(cfg.train.device)

        # Evaluate base model
        print("\nEvaluating base model...")
        base_rank, base_hit = evaluate_model_on_test(base_model, cfg, test_df)
        print(f"[Base] avg_rank_best = {base_rank:.4f}, Hit@K = {base_hit:.4f}")

        # Evaluate fine-tuned model
        print("\nEvaluating fine-tuned model...")
        ft_rank, ft_hit = evaluate_model_on_test(model, cfg, test_df)
        print(f"[Finetuned] avg_rank_best = {ft_rank:.4f}, Hit@K = {ft_hit:.4f}")

        # Show improvement
        rank_improvement = base_rank - ft_rank
        hit_improvement = ft_hit - base_hit
        print(f"\nImprovement: rank ↓{rank_improvement:.4f}, hit ↑{hit_improvement:.4f}")
        return {
            "test_mrr": 1.0 / ft_rank if ft_rank > 0 else 0.0,  # Convert rank to MRR approximation
            "test_hit@10": ft_hit,
            "base_mrr": 1.0 / base_rank if base_rank > 0 else 0.0,
            "base_hit@10": base_hit,
        }

    except Exception as e:
        print(f"Could not evaluate on test set: {e}")
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    print("Loading training and validation data...")
    train_examples, eval_examples = load_train_eval_examples(cfg)

    print(f"Training examples: {len(train_examples):,}")
    print(f"Validation examples: {len(eval_examples):,}")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=cfg.dataset.batching.train_batch_size,
        num_workers=cfg.train.num_workers,
    )

    # Create validation evaluator for monitoring during training
    print("Setting up validation evaluator...")
    val_evaluator = create_validation_evaluator(cfg, eval_examples)

    model = SentenceTransformer(cfg.model.base_model_name)
    if cfg.train.device and cfg.train.device != "cpu":
        model = model.to(cfg.train.device)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    num_train_steps = len(train_dataloader) * cfg.train.num_epochs
    warmup_steps = int(cfg.train.warmup_ratio * num_train_steps)

    output_path = to_absolute_path(cfg.train.output.dir)

    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    print(f"Model: {cfg.model.base_model_name}")
    print(f"Epochs: {cfg.train.num_epochs}")
    print(f"Batch size: {cfg.dataset.batching.train_batch_size}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Output path: {output_path}")
    print("="*50)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        epochs=cfg.train.num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        save_best_model=True,  # Save best model based on validation performance
    )

    print(f"\nTraining finished. Model saved to {output_path}")

    # Final evaluation on test set
    evaluate_on_test_set(cfg, model)


if __name__ == "__main__":
    main()
