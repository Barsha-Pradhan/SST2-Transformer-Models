# SST2 Transformer Models Project

## Description
This project implements multiple Transformer models on the SST2 sentiment analysis dataset using Hugging Face Transformers.

## Models Used
- TinyBERT
- ALBERT
- MobileBERT
- DistilBERT
- BERT-base-uncased

## Dataset
- SST2 (Stanford Sentiment Treebank)

## Results
## Final Model Comparison

| Model | Accuracy | Size (MB) | Latency (ms) |
|---|---|---|---|
| TinyBERT | 0.82 | 54.75 | 119.68 |
| ALBERT | 0.864 | 44.58 | 122.05 |
| MobileBERT | 0.54 | 93.91 | 144.30 |
| DistilBERT | 0.854 | 255.43 | 70.95 |
| BERT-base | 0.876 | 417.67 | 120.57 |

## Tools Used
- Python
- PyTorch
- Hugging Face Transformers
