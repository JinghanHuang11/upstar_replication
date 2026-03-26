#!/bin/bash
# Quick test: Baseline evaluation on small subset
# Verifies evaluation module stability

set -e

echo "============================================================"
echo "Testing Baseline Evaluation Module"
echo "============================================================"

echo ""
echo "This will:"
echo "  1. Train baseline on small subset (1000 users, 10 epochs)"
echo "  2. Evaluate on test set"
echo "  3. Verify metrics are reasonable"
echo ""

# Check prerequisites
if [ ! -f "data/processed/tafeng/metadata.pkl" ]; then
    echo "ERROR: Data not found. Run: bash scripts/run_preprocess.sh"
    exit 1
fi

# Create test config
cat > configs/test_baseline.yaml <<EOF
# Test baseline config (small subset)
dataset:
  name: tafeng
  processed_dir: data/processed/tafeng/

model:
  embed_dim: 64
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2
  max_seq_length: 50

training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.00001
  max_epochs: 10
  early_stop_patience: 3
  metric_for_best: 'NDCG@10'

evaluation:
  k_values: [5, 10, 20]

seed: 42
EOF

# Modify training script to use subset
echo ""
echo "Training baseline on subset..."
echo "------------------------------------------------------------"

python - <<PYTHON_SCRIPT
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append('.')

from src.models.baseline_lstm import LSTMRec as BaselineLSTM
from src.data.dataloader import get_dataloader
from src.evaluation.metrics import compute_all_metrics

# Load config
with open('configs/test_baseline.yaml') as f:
    config = yaml.safe_load(f)

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load metadata
processed_dir = Path(config['dataset']['processed_dir'])
with open(processed_dir / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

num_items = metadata['num_items']
print(f"Dataset: {metadata['num_users']} users, {num_items} items")

# Load sequences
with open(processed_dir / 'train_sequences.pkl', 'rb') as f:
    train_sequences = pickle.load(f)

with open(processed_dir / 'test_sequences.pkl', 'rb') as f:
    test_sequences = pickle.load(f)

print(f"Train users: {len(train_sequences)}")
print(f"Test users: {len(test_sequences)}")

# For quick test, limit to first 1000 users
train_sequences_subset = dict(list(train_sequences.items())[:1000])
test_sequences_subset = dict(list(test_sequences.items())[:1000])

print(f"Using subset: {len(train_sequences_subset)} train, {len(test_sequences_subset)} test")

# Create dataloaders
batch_size = config['training']['batch_size']
max_seq_length = config['model']['max_seq_length']

train_loader = get_dataloader(
    sequences=train_sequences_subset,
    max_seq_length=max_seq_length,
    batch_size=batch_size,
    split='train',
    num_workers=0
)
test_loader = get_dataloader(
    sequences=test_sequences_subset,
    max_seq_length=max_seq_length,
    batch_size=batch_size,
    split='test',
    num_workers=0
)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# Create model
model = BaselineLSTM(
    num_items=num_items,
    embed_dim=config['model']['embed_dim'],
    hidden_dim=config['model']['hidden_dim'],
    num_layers=config['model']['num_layers'],
    dropout=config['model']['dropout']
).to(device)

print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# Train
max_epochs = config['training']['max_epochs']

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        items = batch['items'].to(device)
        seq_length = batch['seq_length'].to(device)
        target_item = batch['target'].to(device)

        optimizer.zero_grad()

        logits = model(items, seq_length)
        loss = criterion(logits, target_item)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    train_loss /= num_batches
    print(f"Epoch {epoch + 1}/{max_epochs}: Train loss = {train_loss:.4f}")

# Evaluate
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        items = batch['items'].to(device)
        seq_length = batch['seq_length'].to(device)
        target_item = batch['target'].to(device)

        logits = model(items, seq_length)
        all_predictions.append(logits.cpu())
        all_targets.append(target_item.cpu())

all_predictions = torch.cat(all_predictions, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Compute metrics
metrics = compute_all_metrics(
    all_predictions, all_targets,
    k_values=[5, 10, 20]
)

print("\n" + "=" * 60)
print("Test Results")
print("=" * 60)
print(f"Precision@5:  {metrics['Precision@5'] * 100:.2f}%")
print(f"Precision@10: {metrics['Precision@10'] * 100:.2f}%")
print(f"Precision@20: {metrics['Precision@20'] * 100:.2f}%")
print(f"NDCG@5:       {metrics['NDCG@5'] * 100:.2f}%")
print(f"NDCG@10:      {metrics['NDCG@10'] * 100:.2f}%")
print(f"NDCG@20:     {metrics['NDCG@20'] * 100:.2f}%")
print(f"MRR@5:        {metrics['MRR@5'] * 100:.2f}%")
print(f"MRR@10:       {metrics['MRR@10'] * 100:.2f}%")
print(f"MRR@20:       {metrics['MRR@20'] * 100:.2f}%")
print("=" * 60)

# Check if reasonable
if metrics['NDCG@10'] > 0.01 and metrics['NDCG@10'] < 0.30:
    print("\n[OK] Metrics are in reasonable range!")
    print("   (NDCG@10 between 1% and 30%)")
elif metrics['NDCG@10'] <= 0.01:
    print("\n[WARNING] NDCG@10 too low (<1%)")
    print("    Model may not be learning properly")
else:
    print("\n[WARNING] NDCG@10 very high (>30%)")
    print("    May indicate data leakage or evaluation bug")

PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Baseline evaluation test complete!"
    echo "============================================================"
else
    echo "Test failed!"
    exit 1
fi
