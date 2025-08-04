import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
from pandas.plotting import autocorrelation_plot
import torch.nn as nn
import torch.optim as optim
import os
import utils
import price_models as models

NUM_MODELS = 3  # Number of models to train
MODEL_NAMES = ['CNN', 'LSTM', 'CNNLSTM']  # Names of the models
NUM_FEATURE_SETS = 6  # Number of feature sets to use


def name2model(modelName, input_dim, output_size = 1):
    if modelName == 'CNNLSTM':
        return models.CNNLSTM(input_size=input_dim, hidden_size=utils.HIDDEN_SIZE, output_size=output_size)
    elif modelName == 'CNN':
        return models.CNN(input_size=input_dim, hidden_size=utils.HIDDEN_SIZE, output_size=output_size)
    elif modelName == 'LSTM':
        return models.LSTM(input_size=input_dim, hidden_size=utils.HIDDEN_SIZE, output_size=output_size)
    else:
        raise ValueError(f"Unknown model name: {modelName}")


def create_sequences(x, y, time_steps=utils.TIME_STEPS):
    X_seq, y_seq = [], []
    for i in range(len(x) - time_steps):
        X_seq.append(x[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def train(model, path, train_loader, test_loader, criterion, optimizer, i, j, scheduler=None, num_epochs=100, early_stopping_patience=10):
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Move model to device
        model.to(device)
        
        # Lists to store metrics
        train_losses = []
        test_losses = []
        
        # Early stopping variables
        min_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                # Move data to device
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    # Move data to device
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    # Forward pass
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            # Print progress
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {avg_train_loss:.6f}, '
                f'Test Loss: {avg_test_loss:.6f}')
            
            # Step the scheduler
            if scheduler:
                scheduler.step(avg_test_loss)

            # Early stopping check
            if avg_test_loss < min_val_loss:
                min_val_loss = avg_test_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Load the best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save the trained model
        model_save_path = os.path.join("result", "close", path)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved LSTM model to {model_save_path}")
        
        return model, train_losses, test_losses

 # Evaluate the model
def evaluate(model, X_test_tensor, y_test_tensor, y_scaler):
    # Set model to evaluation mode
    model.eval()
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_test_tensor = X_test_tensor.to(device)
    
    # Make predictions
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    # Inverse transform to get actual values
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_actual = y_scaler.inverse_transform(y_test_tensor.numpy())
    
    # Calculate metrics
    mse = mean_squared_error(y_actual, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    return y_pred, y_actual, mse, rmse, mae, r2


# Function to make predictions on new data
def predictClosePrice(model, new_data, X_scaler, y_scaler, time_steps=20):
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    # Scale data
    new_data_scaled = X_scaler.transform(new_data)

    # Create sequences
    sequences = np.array([
        new_data_scaled[i:i+time_steps]
        for i in range(len(new_data_scaled) - time_steps + 1)
    ])

    sequences_tensor = torch.FloatTensor(sequences).to(device)

    # Predict
    with torch.no_grad():
        predicted_scaled = model(sequences_tensor).cpu().numpy()

    # Inverse scale predictions
    predictions = y_scaler.inverse_transform(predicted_scaled)

    return predictions.flatten()

def dataPreprocessing(i):
    # Load and preprocess data
    train_df = pd.read_feather("../data/ETHUSDT/df_train.feather")
    test_df  = pd.read_feather("../data/ETHUSDT/df_test.feather")

    train_data = utils.add_time_features(train_df)
    test_data  = utils.add_time_features(test_df)

    # Scale features
    feature_cols = utils.get_feature_cols(i)
    x_scaler, y_scaler, X_train, X_test, y_train, y_test  = utils.scale_data(train_data, test_data, feature_cols, isTrend=False)

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq).view(-1, 1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=utils.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=utils.BATCH_SIZE)

    return train_loader, test_loader, x_scaler, y_scaler, feature_cols, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def pricePrediction(modelName, i, j):
    # Preprocess data
    train_loader, test_loader, x_scaler, y_scaler, feature_cols, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = dataPreprocessing(j)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = name2model(modelName, len(feature_cols))

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=utils.LEARNING_RATE)

    # Train the model
    model_path = f"best_{modelName}_{i+1}_{j+1}_metrics.pth"
    model, train_losses, test_losses = train(
        model=model,
        path=model_path,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        i=i,
        j=j,
        num_epochs=utils.NUM_EPOCHS,
        early_stopping_patience=utils.EARLY_STOPPING_PATIENCE
    )

    # Evaluate the model
    y_pred, y_actual, mse, rmse, mae, r2 = evaluate(
        model=model,
        X_test_tensor=X_test_tensor,
        y_test_tensor=y_test_tensor,
        y_scaler=y_scaler
    )
    
    # Make predictions on test set
    # y_pred_test = predictClosePrice(
    #     model=model,
    #     new_data=X_test_tensor.numpy(),
    #     X_scaler=x_scaler,
    #     y_scaler=y_scaler,
    #     time_steps=utils.TIME_STEPS
    # )
