# Me2U is a Hybrid Gift Recommendation System

Me2U is a scalable hybrid recommendation system that generates personalized gift suggestions using past Amazon product data. It combines collaborative filtering (ALS) with content-based filtering (TF-IDF) and evaluates performance using ranking metrics.


## Dataset
* Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
* Categories used:
  * All_Beauty
  * Amazon_Fashion
  * Appliances
  * Arts_Crafts_and_Sewing

The dataset includes:
* User purchase interactions (filtered to keep verified purchases)
* Product metadata filtered to keep only (title, store, category)

---

## Model Overview

This project implements a **hybrid recommendation system**:

### 1. Collaborative Filtering (ALS)
* Uses implicit feedback (purchase interactions)
* Learns latent embeddings for users and items
* Built with the `implicit` library

### 2. Content-Based Filtering (TF-IDF)
* Extracts item features from:

  * Title
  * Store
  * Category
* Converts text into vector representations

### 3. Hybrid Ranking
* ALS generates candidate items
* TF-IDF re-ranks candidates
* Final score:

  [
  score = \alpha \cdot ALS + (1 - \alpha) \cdot Content
  ]

---

## Evaluation

* Metric: **NDCG@10** (Normalized Discounted Cumulative Gain) based on the top 10 results
* Validation and test sets are created using **sequential user splits**
* Typical performance:
    Inital tests yielded 
  * Validation NDCG@10: ~0.013
  * Test NDCG@10: ~0.013

---

## Project Structure

```bash
.
├── data processing
│   └── CleanMegre.py        # Builds filtered dataset + train/val/test splits
│
├── model
│   └── hybrid_rec_system.py    # ALS + TF-IDF hybrid recommender
│
├── processed/
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
│
└── README.md
```

---

## Installation

Install dependencies:

```bash
pip install pandas numpy scipy scikit-learn pyarrow implicit
```

---

## How to Run

### 1. Download dataset

From the dataset link, download for each category:

* `<category>.jsonl` (reviews)
* `meta_<category>.jsonl` (metadata)

Place all files in the root directory.

---

### 2. Preprocess data

Run:

```bash
python CleanMegre.py
```

This will:

* Filter users/items with low interactions
* Keep only verified purchases
* Merge reviews with metadata
* Build sequential train/validation/test splits
* Save results to `/processed`

---

### 3. Train and evaluate model

```bash
python hybrid_rec_system.py
```

This will:

* Build the user-item interaction matrix
* Train ALS models with different hyperparameters
* Tune:

  * `alpha` (hybrid weight)
  * `factors` (latent dimensions)
  * `regularization`
* Evaluate using NDCG@10
* Output final recommendations

---

## Key Design Choices

* **Interaction filtering**
  * Minimum 5 interactions per user/item
  * Reduces sparsity and improves signal quality

* **Sequential splitting**
  * Uses past interactions to predict future ones
  * Mimics real-world recommendation scenarios

* **Candidate generation + re-ranking**
  * ALS retrieves top 200 candidates
  * TF-IDF refines ranking for personalization

---

## Example Output

```text
BEST PARAMS: (0.5, 128, 0.1)
FINAL TEST NDCG@10: 0.0134

Sample recommendations:
['B0716RCRKG', 'B07BR3DBCL', 'B0BH5MMCWD', ...]
```

---

## Limitations

* Sparse interaction data limits performance
* Implicit feedback lacks strong preference signals
* TF-IDF features are relatively simple

---

## Future Improvements

* Use stronger signals (e.g., ratings, frequency, recency)
* Replace TF-IDF with deep text embeddings
* Incorporate user features
* Optimize ranking with learning-to-rank methods

---

## Author

Jackie Wu
