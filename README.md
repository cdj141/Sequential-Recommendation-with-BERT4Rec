# BERT4Rec Hybrid Recommender System

Sequential recommendation system based on BERT4Rec with hybrid ranking
fusion.

## Overview

This project implements a personalized recommender system using
BERT4Rec, a bidirectional Transformer-based sequential model with masked
item prediction.

To improve robustness under sparse and cold-start conditions, a hybrid
inference pipeline combines three ranking signals:

1.  BERT4Rec contextual predictions
2.  Global item popularity
3.  Title-based semantic similarity (DistilBERT)

Final rankings are generated using Reciprocal Rank Fusion (RRF).

Public leaderboard performance: Recall@10 = 0.0201

## Dataset

Three CSV files are used:

-   train.csv (user-item interactions with timestamps)
-   sample_submission.csv
-   item_meta.csv (item metadata)

Preprocessing steps:

-   Group and sort interactions per user by timestamp
-   Keep most recent 50 items per user
-   Left-pad shorter sequences with zeros
-   Apply 15% masked item prediction strategy
-   Reindex item IDs into dense integer space
-   Filter metadata items not present in training data

## Model Architecture

Backbone: BERT4Rec (Transformer encoder)

Components:

-   Item embedding (dimension 512)
-   Positional embedding
-   3 Transformer encoder layers
-   4 attention heads per layer
-   Hidden size 512
-   Dropout 0.1
-   Maximum sequence length 50

Loss function: Cross-entropy loss over masked positions (ignore padding)

Training configuration:

-   Optimizer: Adam
-   Learning rate: 1e-3
-   Batch size: 128
-   Best checkpoint: epoch 3

## Inference Pipeline

1.  BERT4Rec Prediction
    -   Mask up to 15% of sequence
    -   Predict masked items
    -   Remove previously seen items
2.  Popularity Ranking
    -   Rank items by global interaction frequency
3.  Title Similarity Ranking
    -   Compute DistilBERT embeddings for item titles
    -   Use cosine similarity with user's penultimate item
4.  Reciprocal Rank Fusion (RRF)

RRF score: RRF(d) = sum(1 / (k + rank_i(d))), k = 60

## Results

Recall@10 = 0.0201 (public leaderboard)

Observations:

-   Longer user histories improve performance
-   Popularity dominates for short-history users
-   Hybrid fusion improves robustness over single-source ranking

## Limitations

-   No user-side features
-   No separate validation split
-   Static hyperparameters
-   Metadata quality issues (missing/noisy titles)

## Future Improvements

-   Validation-based early stopping
-   Adaptive or query-aware fusion strategies
-   Domain-specific fine-tuning of title encoder
-   Incorporating user-side features

## Technologies

Python PyTorch Transformers NumPy Pandas

## Author

Dongjie Chen MSc Computer Science (Data Science), Leiden University
