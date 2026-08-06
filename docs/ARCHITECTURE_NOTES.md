# Architecture Notes & Developer Guide

## Real-Time Serving
FastAPI endpoints must maintain sub-100ms latency for fraud detection to ensure transaction processing is not bottlenecked. Current average is ~86ms.

## Class Imbalance (Fraud)
To handle the 0.17% fraud rate, SMOTE (Synthetic Minority Over-sampling Technique) is applied exclusively within the cross-validation folds to prevent data leakage.

