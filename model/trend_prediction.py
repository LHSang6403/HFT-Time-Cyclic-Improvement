import utils as utils
import pandas as pd
import numpy as np
import os
import torch
import torch.optim as optim
import trend_models as models
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score

NUM_MODELS = 3  # Number of models to train
MODEL_NAMES = ['CNN', 'LSTM', 'CNNLSTM']  # Names of the models
NUM_FEATURE_SETS = 6  # Number of feature sets to use

# --- Load & preprocess ---
def dataPreprocessing(i):
    train_df = pd.read_feather("../data/ETHUSDT/df_train.feather")
    test_df  = pd.read_feather("../data/ETHUSDT/df_test.feather")

    df_train = utils.add_time_features(train_df)
    df_test  = utils.add_time_features(test_df)
    trend_train = utils.compute_trend(df_train)
    trend_test  = utils.compute_trend(df_test)

    # --- Scale data ---
    feature_cols = utils.get_feature_cols(i)
    scaler, y_scaler, X_train_vals, X_test_vals, _, _ = utils.scale_data(df_train, df_test, feature_cols)
    X_train_seq, y_train_seq = utils.create_sequences(X_train_vals, trend_train)
    X_test_seq,  y_test_seq  = utils.create_sequences(X_test_vals, trend_test)

    # --- Imbalance handling via sampler ---
    class_counts = np.bincount(y_train_seq) 
    sample_w = 1.0 / class_counts[y_train_seq]
    sampler = WeightedRandomSampler(
        weights=sample_w,
        num_samples=len(sample_w),
        replacement=True
    )

    torch_train = TensorDataset(
        torch.FloatTensor(X_train_seq),
        torch.LongTensor(y_train_seq)
    )
    torch_test = TensorDataset(
        torch.FloatTensor(X_test_seq),
        torch.LongTensor(y_test_seq)
    )

    train_loader = DataLoader(
        torch_train,
        batch_size=utils.BATCH_SIZE,
        sampler=sampler,
        drop_last=True
    )
    test_loader = DataLoader(
        torch_test,
        batch_size=utils.BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    return train_loader, test_loader, scaler, feature_cols

def name2model(modelName, feature_cols, device):
    if modelName == 'CNNLSTM':
        return models.CNNLSTM(len(feature_cols), device)
    elif modelName == 'CNN':
        return models.CNN(len(feature_cols), device)
    elif modelName == 'LSTM':
        return models.LSTM(len(feature_cols), device)
# --- Focal loss (no smoothing, no weight) ---
# Using for handling hard patterns in trend prediction
def focal_loss(inputs, targets, gamma=2.0):
    ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()

def train(model, device, path, train_loader, test_loader, optimizer, scheduler, i, j):
    best_loss = float('inf')
    for epoch in range(1, utils.NUM_EPOCHS+1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = focal_loss(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), utils.GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += focal_loss(model(xb), yb).item()
        val_loss /= len(test_loader)

        print(f"Epoch {epoch}/{utils.NUM_EPOCHS} – "
            f"Train loss {total_loss/len(train_loader):.6f}, "
            f"Val loss   {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            model_save_path = os.path.join("result", "trend", path)
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved LSTM model to {model_save_path}")

        # Early stopping
        if epoch > utils.EARLY_STOPPING_PATIENCE and val_loss > best_loss:
            print(f"Early stopping at epoch {epoch}")
            break

    print("Training finished.")


def evaluate(model, device, path, test_loader):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            labels.extend(yb.numpy())

    # Direction accuracy
    print("classification accuracy:", accuracy_score(labels, preds))
    print(labels)
    # Precision & Recall cho từng class: [0=flat, 1=up, 2=down]
    precision_per_class = precision_score(labels, preds, labels=[0,1,2], average=None, zero_division=0)
    recall_per_class    = recall_score   (labels, preds, labels=[0,1,2], average=None, zero_division=0)
    print(f"Precision per class [flat, up, down]: {precision_per_class}")
    print(f"Recall    per class [flat, up, down]: {recall_per_class}")

def trendPrediction(modelName, i, j):
    # --- Load & preprocess data ---
    train_loader, test_loader, scaler, feature_cols = dataPreprocessing(j)

    # --- Model, optimizer, scheduler ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = name2model(modelName, feature_cols, device)

    # Sử dụng AdamW để cập nhật trọng số.
    # Tự động điều chỉnh learning rate theo chu kỳ để tăng hiệu quả học.
    optimizer = optim.AdamW(model.parameters(),
                            lr=utils.LEARNING_RATE,
                            weight_decay=utils.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10
    )
    model.to(device)
    os.makedirs('result/trend', exist_ok=True)

    modelPath = f'result/trend/best_{modelName.lower()}_{i+1}_{j+1}_metrics.pth'

    # --- Train & evaluate ---
    train(
        model=model,
        device=device,
        path=modelPath,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        i=i,
        j=j
    )

    evaluate(
        model=model,
        device=device,
        path=f'result/trend/{modelPath}',
        test_loader=test_loader
    )
