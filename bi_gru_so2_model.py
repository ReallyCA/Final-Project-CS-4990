import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import timedelta, datetime
import csv

# AQI breakpoints for multiple pollutants
AQI_BREAKPOINTS = {
    'PM25': [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500)
    ],
    'PM10': [
        (0,   54,   0,   50),
        (55, 154,  51,  100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500)
    ],
    'Ozone': [
        (0.000, 0.054,   0,   50),
        (0.055, 0.070,  51,  100),
        (0.071, 0.085, 101,  150),
        (0.086, 0.105, 151,  200),
        (0.106, 0.200, 201, 300)
    ],
    'CO': [
        (0.0, 4.4,   0,   50),
        (4.5, 9.4,  51,  100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300)
    ],
    'SO2': [
        (0,   35,   0,   50),
        (36,  75,  51,  100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 500)
    ],
    'NO2': [
        (0,   53,   0,   50),
        (54, 100,  51,  100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500)
    ]
}

def calculate_aqi_from_conc(conc: float, pollutant: str) -> float:
    if pd.isna(conc):
        return np.nan
    if pollutant not in AQI_BREAKPOINTS:
        raise ValueError(f"No breakpoints defined for {pollutant}")
    for low, high, aqi_low, aqi_high in AQI_BREAKPOINTS[pollutant]:
        if low <= conc <= high:
            return aqi_low + (aqi_high - aqi_low) * (conc - low) / (high - low)
    return 500.0

def load_epa_hourly_data(filepaths,
                         state: str = 'California',
                         pollutant: str = 'Ozone') -> pd.DataFrame:
    """
    Load and combine multiple EPA hourly CSV files into one DataFrame.
    filepaths: list of CSV file paths
    """
    counties = ["Los Angeles", "Orange", "San Francisco"]

    dfs = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, engine="python", on_bad_lines="skip")

        # Filter for pollutant in target state and counties
        df = df[(df['State Name'] == state) &
                (df['Parameter Name'] == pollutant) &
                (df['County Name'].isin(counties))]

        # Restrict to POC == 1 for consistency
        df = df[df['POC'] == 1]

        # Build datetime index
        df['datetime'] = pd.to_datetime(df['Date Local'] + ' ' + df['Time Local'])
        df = df.sort_values('datetime').set_index('datetime')

        # Keep concentration + county name
        df = df[['County Name', 'Sample Measurement']].rename(
            columns={'Sample Measurement': pollutant}
        )

        # Clean invalids
        df[pollutant] = df[pollutant].replace([np.inf, -np.inf], np.nan)
        df.loc[df[pollutant] < 0, pollutant] = np.nan

        # Compute AQI
        df['AQI'] = df[pollutant].apply(lambda x: calculate_aqi_from_conc(x, pollutant))

        dfs.append(df)

    # Concatenate all years together
    combined_df = pd.concat(dfs).sort_index()
    return combined_df


def safe_preprocess(df: pd.DataFrame, pollutant: str):
    valid_frames = []
    scalers = {}

    for county, group in df.groupby("County Name"):
        if group[pollutant].dropna().shape[0] == 0:
            print(f"Skipping {county}: no samples found")
            continue

        # Interpolate missing values (linear only to avoid datetime issues)
        group[pollutant] = group[pollutant].interpolate(method='linear')
        group[pollutant] = group[pollutant].ffill().bfill()

        # Compute AQI
        group['AQI'] = group[pollutant].apply(lambda x: calculate_aqi_from_conc(x, pollutant))

        # Scale pollutant values
        scaler = StandardScaler()
        group[[pollutant]] = scaler.fit_transform(group[[pollutant]])
        scalers[county] = scaler

        valid_frames.append(group)

    if not valid_frames:
        raise ValueError("No counties with valid samples found!")

    df_clean = pd.concat(valid_frames).sort_index()
    return df_clean, scalers

def create_sequences(df: pd.DataFrame, pollutant: str, seq_length: int = 24):
    values = df[[pollutant, 'AQI']].values
    xs, ys = [], []
    for i in range(len(values) - seq_length):
        x = values[i:i+seq_length, 0:1]  # pollutant history
        y = values[i+seq_length, 1]      # AQI next hour
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def time_series_train_test_split(X, y, test_ratio=0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


class BiGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


def train_model(X_train, y_train,
                epochs=50, batch_size=32, lr=1e-3,
                hidden_dim=128, num_layers=2, dropout=0.3,
                device='cpu', save_path="model.pth"):
    model = BiGRU(input_dim=1, hidden_dim=hidden_dim,
                  num_layers=num_layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        avg_loss = epoch_loss / len(train_ds)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model


def evaluate_model(model, X_test, y_test, device='cpu'):
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_t).cpu().numpy()
    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    return y_pred

def forecast_future(model, df, pollutant: str,
                    seq_length=24, horizon=24,
                    device='cpu', save_path=None):
    """
    Generate AQI predictions for multiple future hours autoregressively
    and save them into a timestamped CSV file if save_path is provided.
    """
    context = df.iloc[-seq_length:][[pollutant]].values
    context_t = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    preds = []
    current_dt = df.index[-1]

    for step in range(horizon):
        with torch.no_grad():
            next_aqi = model(context_t).item()
        current_dt += timedelta(hours=1)
        preds.append((current_dt, next_aqi))

        # Update context autoregressively
        next_pollutant_val = context_t[0, -1, 0].item()
        new_seq = context_t[0, 1:, 0].cpu().numpy().tolist()
        new_seq.append(next_pollutant_val)
        context_t = torch.tensor(new_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    # Print results
    for dt, aqi in preds:
        print(f"Predicted AQI for {pollutant} at {dt}: {aqi:.2f}")

    # Save to CSV with timestamp
    if save_path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{save_path.rstrip('.csv')}_{timestamp}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Datetime", f"{pollutant}_Predicted_AQI"])
            for dt, aqi in preds:
                writer.writerow([dt, aqi])
        print(f"Forecast saved to {filename}")

    return preds

def load_model(checkpoint_path: str,
               input_dim=1, hidden_dim=128, num_layers=2, dropout=0.3,
               device='cpu'):
    """
    Reload a saved BiGRU model checkpoint.
    """
    model = BiGRU(input_dim=input_dim, hidden_dim=hidden_dim,
                  num_layers=num_layers, dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model reloaded from {checkpoint_path}")
    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Multiple years of data for Ozone
    filepaths = [
        "so2data/so2_epa_hourly_23.csv",
        "so2data/so2_epa_hourly_24.csv",
        "so2data/so2_epa_hourly_25.csv"
    ]
    pollutant = "SO2"

    df = load_epa_hourly_data(filepaths, state="California", pollutant=pollutant)
    df_clean, scalers = safe_preprocess(df, pollutant)

    print(f"Samples: {len(df_clean)} | Date range: {df_clean.index.min()} — {df_clean.index.max()}")
    print(f"Counties included: {df_clean['County Name'].unique()}")

    # Build sequences
    X, y = create_sequences(df_clean, pollutant=pollutant, seq_length=24)
    X_train, y_train, X_test, y_test = time_series_train_test_split(X, y, test_ratio=0.2)

    # Train or reload model
    checkpoint_path = f"base_models/{pollutant.lower()}_bigru_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Found existing model for {pollutant}, loading...")
        model = load_model(checkpoint_path, device=device)
    else:
        print(f"No model found for {pollutant}, training new one...")
        model = train_model(X_train, y_train, epochs=50, batch_size=32, lr=1e-3,
                            hidden_dim=128, num_layers=2, dropout=0.3,
                            device=device, save_path=checkpoint_path)

    # Evaluate
    evaluate_model(model, X_test, y_test, device)

    # Forecast next 24 hours and save to CSV
    forecast_future(model, df_clean, pollutant=pollutant,
                    seq_length=24, horizon=24,
                    device=device, save_path=f"base_gen_AQI/{pollutant.lower()}_forecast.csv")


if __name__ == "__main__":
    main()