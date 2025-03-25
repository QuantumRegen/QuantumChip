import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import time


# Insane data
def generate_stock_data(n_rows=100_000_000):
    print(f"Generating {n_rows:,} rows of stock data...")
    dates = pd.date_range("2020-01-01", periods=n_rows // 1000, freq="S")
    prices = np.random.uniform(50, 150, n_rows) + np.sin(np.linspace(0, 200, n_rows)) * 20
    data = pd.DataFrame({
        "Date": np.tile(dates, 1000),
        "Price": prices,
        "Volume": np.random.randint(100, 1000000, n_rows),
        "MA10": pd.Series(prices).rolling(10).mean().fillna(prices[0]),
        "MA50": pd.Series(prices).rolling(50).mean().fillna(prices[0]),
    })
    data["Next_Price"] = data["Price"].shift(-1).fillna(data["Price"].iloc[-1])
    return data


# Prep data
def prepare_data(data):
    features = data[["Price", "Volume", "MA10", "MA50"]].values
    target = data["Next_Price"].values.reshape(-1, 1)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32)
    return X, y


# Monster net
class StockMelter(nn.Module):
    def __init__(self, input_size=4):
        super(StockMelter, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x


# Train ‘til it hurts
def train_model(X, y, epochs=50, batch_size=8192):
    model = StockMelter()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {X.shape[0]:,} samples, {epochs} epochs...")
    for epoch in range(epochs):
        start_time = time.time()
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Time: {epoch_time:.2f}s")
    return model


# Main
if __name__ == "__main__":
    start_time = time.time()

    # Data
    stock_data = generate_stock_data()
    print(f"Data size: {stock_data.memory_usage().sum() / 1024 ** 2:.2f} MB")
    X, y = prepare_data(stock_data)

    # Train
    model = train_model(X, y)

    # Predict (sample)
    with torch.no_grad():
        sample_X = X[:10]
        preds = model(sample_X)
        print("\nSample Predictions (Next Price):")
        for i, (actual, pred) in enumerate(zip(y[:10], preds)):
            print(f"Sample {i + 1}: Actual {actual.item():.2f}, Predicted {pred.item():.2f}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")