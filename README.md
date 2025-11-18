# ETHAIM: E-commerce Product Search with Biencoder Model

## Overview

ETHAIM is an e-commerce product search system that uses a **biencoder architecture** based on the multilingual E5 model to learn semantic embeddings for queries and products. The system fine-tunes pre-trained language models to optimize retrieval performance for product search scenarios.

## Architecture

### Biencoder Model

The system implements a **biencoder architecture** where:
- **Query Encoder**: Encodes search queries into dense vector representations
- **Product Encoder**: Encodes product information into dense vector representations
- **Retrieval**: Uses cosine similarity to find the most relevant products for a given query

Both encoders share the same transformer backbone (multilingual-e5-base) but encode different types of text with task-specific prefixes:
- Queries: prefixed with `"query: "`
- Products: prefixed with `"passage: "`

## Training Logic

### Data Preparation

1. **Input Data**: Parquet file containing query-product interaction data with columns:
   - `OriginalQuery`: User search queries
   - `ProductName`: Product names
   - `BrandName`: Product brands
   - `ProductType`: Product categories
   - `Level4_ProductGroup`: Detailed product groups
   - `UniqueImpressions`: Number of times product was shown
   - `UniqueSelects`: Number of times product was selected

2. **Data Filtering**:
   - Minimum impressions: ≥1
   - Minimum selects: ≥1
   - Maximum samples: 300,000 (configurable)

3. **Text Construction**:
   - **Query text**: `"query: {OriginalQuery}"`
   - **Product text**: `"passage: {ProductName} | {BrandName} | {Level4_ProductGroup/ProductType}"`

### Training Process

1. **Model Loading**: Initialize with `intfloat/multilingual-e5-base`

2. **Loss Function**: `MultipleNegativesRankingLoss`
   - Treats each query-product pair as a positive example
   - Uses in-batch negatives for contrastive learning
   - Optimizes cosine similarity between positive pairs

3. **Training Configuration**:
   - Batch size: 256
   - Epochs: 1
   - Warmup ratio: 0.1
   - Optimizer: Adam (default from sentence-transformers)

4. **Data Splitting**:
   - **Recommended**: Predefined splits (80/10/10 train/val/test)
   - **Legacy**: Random split (90/10 train/eval)
   - Test set used for final model evaluation

### Key Training Components

```python
# Loss function for biencoder contrastive learning
train_loss = losses.MultipleNegativesRankingLoss(model)

# Training with positive pairs and in-batch negatives
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=cfg.train.num_epochs,
    warmup_steps=warmup_steps,
)
```

## Evaluation Metrics

### Offline Evaluation

- **Mean Reciprocal Rank (MRR)**: Average rank of the best product for each query
- **Hit@K**: Percentage of queries where the correct product appears in top-K results
- **Evaluation Protocol**:
  - Group products by query
  - For each query, rank all associated products by cosine similarity
  - Measure rank of the product with highest selection count

### Evaluation Process

1. Encode all queries and products separately
2. Compute cosine similarity matrix
3. Rank products for each query
4. Calculate MRR and Hit@10 metrics

## Configuration Structure

```
conf/
├── config.yaml           # Main configuration with defaults
├── dataset/
│   └── embedding_query2product.yaml  # Dataset fields and sampling
├── model/
│   └── e5_base.yaml      # Model architecture and prefixes
├── train/
│   └── base.yaml         # Training hyperparameters
└── eval/
    └── offline.yaml      # Evaluation settings
```

## Usage

### 1. Data Splitting (One-time setup)

Create query-level stratified train/val/test splits to avoid data leakage:

```bash
cd /home/c/yuqyan/code/ETHAIM
python src/split_dataset.py
```

**Key Features:**
- **Query-level splitting**: Each query appears in exactly one split (no leakage)
- **Stratified sampling**: Maintains similar language and frequency distributions across splits
- **Frequency bands**: Queries categorized as Head (top 10%), Mid (middle 30%), Tail (bottom 60%) by total selections

**Output:**
- `./data/splits/train.parquet` (~80% queries)
- `./data/splits/val.parquet` (~10% queries)
- `./data/splits/test.parquet` (~10% queries)

**Verification:**
After splitting, verify no query leakage:
```bash
python verify_split.py
```

### 2. Training with Validation Monitoring & Wandb Logging

**New integrated workflow**: Training now includes validation monitoring, wandb logging, and final test evaluation.

```bash
# Optional: Setup wandb first
bash setup_wandb.sh

# Start training
python src/train_e5_contrastive.py
```

**What happens during training:**
- ✅ **Validation monitoring**: Every epoch evaluates on validation set (MRR, Hit@10)
- ✅ **Wandb logging**: All metrics automatically logged to wandb
- ✅ **Best model saving**: Automatically saves the best performing model on validation set
- ✅ **Final test evaluation**: After training completes, automatically evaluates on test set

**Console output during training:**
```
Wandb initialized: e5-finetune-42

Epoch: 1/2 | Loss: 0.25 | val_mrr: 0.1234 | val_hit@10: 0.0567
Epoch: 2/2 | Loss: 0.18 | val_mrr: 0.1456 | val_hit@10: 0.0678

==================================================
FINAL EVALUATION ON TEST SET
==================================================

Test Set Results:
[Base] avg_rank_best = 45.67, Hit@K = 0.1234
[Finetuned] avg_rank_best = 23.45, Hit@K = 0.2345
Improvement: rank ↓22.22, hit ↑0.1111
```

### Wandb Dashboard

**Project**: ETHAIM (entity: yangyuqi2020-uzh)

**Logged metrics**:
- Training loss (automatic)
- Validation MRR & Hit@10 (per epoch)
- Final test results (at end)
- Model configuration & hyperparameters

**Dashboard URL**: https://wandb.ai/yangyuqi2020-uzh/ETHAIM

### 3. Legacy Evaluation Scripts (Optional)

If you need separate evaluation:

```bash
# Validation evaluation (uses random query sampling)
python src/eval_e5_offline.py

# Test set evaluation (uses predefined test split)
python src/eval_test.py
```

### 5. Basic Testing

```bash
python test_e5.py
```

## Data Splitting Strategy

### Query-Level Splitting (Critical)

**Problem:** Traditional row-level splitting allows the same query to appear in both training and test sets, causing **data leakage**. The model might memorize query-product associations rather than learning semantic matching.

**Solution:** Split at the **query level** - each `OriginalQuery` appears in exactly one split (train/val/test).

**Impact:** Evaluation now properly measures: *"Can the model retrieve relevant products for completely new queries it has never seen during training?"*

### Stratified Sampling by Language & Frequency

**Why stratification matters:**
- **Language balance**: E-commerce queries come in multiple languages (de, fr, en, etc.)
- **Frequency distribution**: Query popularity follows power-law distribution
- **Realistic evaluation**: Test set should represent the same query distribution as production

**Implementation:**
- **Language**: Stratify by query language (mode of PageLanguage per query)
- **Frequency bands**:
  - **Head**: Top 10% queries by total selections
  - **Mid**: Middle 30% queries
  - **Tail**: Bottom 60% queries (long tail)
- **Combined stratification**: `(language, frequency_band)` combinations

**Benefits:**
- Each split maintains representative language and popularity distributions
- Prevents bias toward high-frequency queries in test set
- Ensures robust performance across different query types

## Key Design Decisions

### Why Biencoder Architecture?

1. **Efficiency**: Separate encoding allows pre-computing product embeddings for fast retrieval
2. **Scalability**: Can handle large product catalogs without re-encoding queries
3. **Multilingual**: E5 model's multilingual capabilities support international e-commerce

### Why Contrastive Learning?

1. **No Labels Needed**: Uses implicit supervision from user interactions
2. **Robust**: Learns from relative preferences rather than absolute ratings
3. **Generalizable**: Captures semantic similarity beyond exact keyword matches

### Data Sampling Strategy

1. **Interaction-based Filtering**: Only includes products with actual user engagement
2. **Controlled Scale**: Limits to 300K samples for manageable training time
3. **Query Diversity**: Ensures broad coverage of search intents

## Model Artifacts

- **Input**: Fine-tuned SentenceTransformer model
- **Output**: Dense embeddings (768 dimensions) for queries and products
- **Storage**: Saved in `./models/e5-finetuned-v1/`

## Future Improvements

1. **Hard Negative Mining**: Include more challenging negative examples
2. **Curriculum Learning**: Progressive difficulty during training
3. **Multi-task Learning**: Joint optimization with classification objectives
4. **Domain Adaptation**: Fine-tuning on specific product categories
