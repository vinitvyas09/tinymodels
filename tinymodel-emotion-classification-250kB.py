#!/usr/bin/env python3
import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
import cProfile
import pstats
import io
import time
import random

###############################################################################
#                           CONFIGURATION MACROS
###############################################################################
NUM_EPOCHS = 100     # Adjust number of epochs as desired
ENABLE_LOGS = False  # Set to False to reduce logging

def log(message):
    """Logs a message only if ENABLE_LOGS is True."""
    if ENABLE_LOGS:
        print(message)

###############################################################################
#                           HYPERPARAMETERS
###############################################################################
VOCAB_SIZE = 2000    # Limit vocab size
MAX_LENGTH = 20       # Sequence length for padding/truncation
EMBED_DIM = 32        # Embedding dimension
LEARNING_RATE = 0.001 # Learning rate for optimizer
BATCH_SIZE = 512      # Batch size for training

###############################################################################
#                           DOWNLOAD DATASET
###############################################################################
# Two files: train_150k.txt and test_62k.txt
# Format: feeling \t text
TRAIN_URL = "https://raw.githubusercontent.com/cblancac/SentimentAnalysisBert/main/data/train_150k.txt"
TEST_URL = "https://raw.githubusercontent.com/cblancac/SentimentAnalysisBert/main/data/test_62k.txt"

log("[INFO] Downloading the Twitter Sentiment dataset...")
if not os.path.exists("train_150k.txt"):
    with open("train_150k.txt", "wb") as f:
        f.write(requests.get(TRAIN_URL).content)

if not os.path.exists("test_62k.txt"):
    with open("test_62k.txt", "wb") as f:
        f.write(requests.get(TEST_URL).content)

###############################################################################
#                           LOAD & PREPROCESS DATA
###############################################################################
log("[INFO] Loading training data...")
with open("train_150k.txt", "r", encoding="utf-8") as f:
    train_lines = f.read().strip().split("\n")

# Each line: "label<TAB>text"
train_data = []
for line in train_lines:
    parts = line.split("\t", 1)
    if len(parts) == 2:
        label, text = parts
        label = int(label.strip())
        text = text.strip()
        train_data.append((text, label))

# Shuffle data
random.seed(42)
random.shuffle(train_data)

# Split into train and validation
# About 150k lines total, take 120k for train and 30k for validation
train_size = 120000
val_size = 30000
train_subset = train_data[:train_size]
val_subset = train_data[train_size:train_size+val_size]

log(f"[INFO] Training set size: {len(train_subset)}, Validation set size: {len(val_subset)}")

log("[INFO] Loading test data...")
with open("test_62k.txt", "r", encoding="utf-8") as f:
    test_lines = f.read().strip().split("\n")

test_data = []
for line in test_lines:
    parts = line.split("\t", 1)
    if len(parts) == 2:
        label, text = parts
        label = int(label.strip())
        text = text.strip()
        test_data.append((text, label))

log(f"[INFO] Test set size: {len(test_data)}")

###############################################################################
#                           BUILD VOCABULARY
###############################################################################
# We'll build a small vocabulary from the training set.
log("[INFO] Building vocabulary...")
all_words = []
for text, _ in train_subset:
    all_words.extend(text.lower().split())

counter = Counter(all_words)
vocab = [w for w, _ in counter.most_common(VOCAB_SIZE)]
word_to_id = {w: i+1 for i, w in enumerate(vocab)}  # 0 for padding

log(f"\t[INFO] Using vocab_size={VOCAB_SIZE}, max_length={MAX_LENGTH}")

def text_to_ids(text):
    tokens = text.lower().split()
    ids = [word_to_id.get(t, 0) for t in tokens[:MAX_LENGTH]]
    if len(ids) < MAX_LENGTH:
        ids += [0]*(MAX_LENGTH - len(ids))
    return ids

def encode_dataset(data):
    X = np.array([text_to_ids(d[0]) for d in data], dtype=np.int64)
    Y = np.array([d[1] for d in data], dtype=np.int64)
    return X, Y

log("[INFO] Encoding datasets...")
X_train, Y_train = encode_dataset(train_subset)
X_val, Y_val = encode_dataset(val_subset)
X_test, Y_test = encode_dataset(test_data)

log(f"\t[INFO] X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
log(f"\t[INFO] X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
log(f"\t[INFO] X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

###############################################################################
#                           DEFINE MODEL
###############################################################################
log("[INFO] Defining a tiny PyTorch model for Sentiment Analysis...")
class TinySentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, max_length=MAX_LENGTH):
        super(TinySentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size+1, embed_dim, padding_idx=0)
        self.linear = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        emb = self.embedding(x)             # (batch, max_length, embed_dim)
        avg_emb = emb.mean(dim=1)           # (batch, embed_dim)
        logits = self.linear(avg_emb)       # (batch, 1)
        probs = self.sigmoid(logits)        # (batch, 1)
        return probs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinySentimentModel(vocab_size=VOCAB_SIZE).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
batch_size = BATCH_SIZE

X_train_t = torch.tensor(X_train, device=device)
Y_train_t = torch.tensor(Y_train, device=device, dtype=torch.float32)
X_val_t = torch.tensor(X_val, device=device)
Y_val_t = torch.tensor(Y_val, device=device, dtype=torch.float32)
X_test_t = torch.tensor(X_test, device=device)
Y_test_t = torch.tensor(Y_test, device=device, dtype=torch.float32)

###############################################################################
#                           EVALUATION FUNCTION
###############################################################################
def evaluate_accuracy(model, X_data, Y_data):
    model.eval()
    with torch.no_grad():
        probs = model(X_data).squeeze()
        preds = (probs >= 0.5).long()
        correct = (preds == Y_data.long()).sum().item()
        return correct / len(Y_data)

###############################################################################
#                           INITIAL ACCURACY
###############################################################################
log("[INFO] Evaluating initial test accuracy (before training)...")
initial_acc = evaluate_accuracy(model, X_test_t, Y_test_t)
log(f"\tInitial Test Accuracy (before training): {initial_acc:.3f}")

###############################################################################
#                           TRAINING FUNCTION
###############################################################################
def train_model():
    model.train()
    for epoch in range(NUM_EPOCHS):
        if ENABLE_LOGS:
            print(f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS} starting...")

        start_time = time.time()

        # Shuffle data
        perm = torch.randperm(len(X_train_t))
        X_train_shuffled = X_train_t[perm]
        Y_train_shuffled = Y_train_t[perm]

        total_loss = 0.0
        batches = 0
        for i in range(0, len(X_train_shuffled), batch_size):
            x_batch = X_train_shuffled[i:i+batch_size]
            y_batch = Y_train_shuffled[i:i+batch_size].unsqueeze(1)

            optimizer.zero_grad()
            probs = model(x_batch)
            loss = criterion(probs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        epoch_time = time.time() - start_time
        avg_loss = total_loss / batches
        if ENABLE_LOGS:
            print(f"\t[TRAIN] Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")

###############################################################################
#                           PROFILE TRAINING
###############################################################################
log("[INFO] Profiling the training process...")
pr = cProfile.Profile()
pr.enable()

start_train_time = time.time()
train_model()
end_train_time = time.time()

pr.disable()

# Print profiling results if logs are enabled
if ENABLE_LOGS:
    log("[INFO] Profiling Results:")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    print(s.getvalue())

###############################################################################
#                           FINAL EVALUATION
###############################################################################
log("[INFO] Evaluating test accuracy after training...")
final_acc = evaluate_accuracy(model, X_test_t, Y_test_t)
log(f"\tFinal Test Accuracy (after training): {final_acc:.3f}")

###############################################################################
#                           SAVE MODEL & REPORT SIZE
###############################################################################
log("[INFO] Saving the tiny model to disk...")
torch.save(model.state_dict(), "tiny_sentiment_model.pt")
model_size = os.path.getsize("tiny_sentiment_model.pt")
log(f"\tModel size in bytes: {model_size}")

###############################################################################
#                           FINAL SUMMARY
###############################################################################
print("\n================ FINAL SUMMARY ================")
print(f"Initial Test Accuracy: {initial_acc:.3f}")
print(f"Final Test Accuracy: {final_acc:.3f}")
print(f"Total Training Time: {end_train_time - start_train_time:.2f} seconds")
print(f"Model Size: {model_size} bytes")
print("===============================================")
