# Architecture Notes & Developer Guide

## Real-Time Serving
FastAPI endpoints must maintain sub-100ms latency for fraud detection to ensure transaction processing is not bottlenecked. Current average is ~86ms.

