# How to Use

## Run the CLI. For example:
• **To simulate an ECG signal:**

```python
python main.py simulate --fs 500 --duration 60
```

• **To train a DL model (for instance, LSTMClassifier):**

```python
python main.py train-dl --model-name LSTMClassifier --num-epochs 10 --batch-size 32
```

• **To train a GNN model:**

```python
python main.py train-gnn --model-name Hybrid_GNN_LSTM --num-epochs 10 --batch-size 32
```