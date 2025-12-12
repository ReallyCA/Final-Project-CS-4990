import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MultiPollutantAQIDataset(Dataset):
    def __init__(self, file_paths):
        self.data = self.load_and_stack(file_paths)

    def load_and_stack(self, file_paths):
        pollutant_arrays = []

        for fp in file_paths:
            df = pd.read_csv(fp)
            aqi_col = [c for c in df.columns if "AQI" in c][0]
            arr = df[aqi_col].values.astype(np.float32)

            if len(arr) != 24:
                raise ValueError(f"File {fp} does not contain 24 hourly values.")

            pollutant_arrays.append(arr)

        stacked = np.stack(pollutant_arrays, axis=0)
        flattened = stacked.reshape(-1)
        return np.array([flattened])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])
    
class JointAQIVAE(nn.Module):
    def __init__(self, input_dim=144, latent_dim=8, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, num_samples=1, latent_dim=8):
        z = torch.randn(num_samples, latent_dim)
        with torch.no_grad():
            recon = self.decode(z)
        return recon

def vae_loss(x, recon, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae(model, dataloader, epochs=200, lr=1e-3, beta=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.float()
            optimizer.zero_grad()

            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(batch, recon, mu, logvar, beta)

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}")

def train_joint_vae(file_paths):
    dataset = MultiPollutantAQIDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = JointAQIVAE(input_dim=144, latent_dim=8, hidden_dim=128)
    train_vae(model, dataloader, epochs=200, lr=1e-3, beta=1.0)

    return model

def generate_year_of_aqi(model, latent_dim=8):
    samples = model.sample(num_samples=365, latent_dim=latent_dim)
    samples = samples.numpy()
    samples = samples.reshape(365, 6, 24)
    return samples

def flatten_to_hourly_csv_with_total(samples):
    rows = []

    for day_idx in range(365):
        for hour in range(24):
            co   = samples[day_idx, 0, hour]
            no2  = samples[day_idx, 1, hour]
            o3   = samples[day_idx, 2, hour]
            pm25 = samples[day_idx, 3, hour]
            pm10 = samples[day_idx, 4, hour]
            so2  = samples[day_idx, 5, hour]

            total_aqi = max(co, no2, o3, pm25, pm10, so2)

            row = {
                "Day": day_idx + 1,
                "Hour": hour,
                "CO": co,
                "NO2": no2,
                "O3": o3,
                "PM25": pm25,
                "PM10": pm10,
                "SO2": so2,
                "Total_AQI": total_aqi
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df

filepaths = {
    "base_gen_AQI/carbon monoxide_forecast_2025-12-10_10-26-59.csv",
    "base_gen_AQI/ozone_forecast_2025-12-09_23-55-08.csv",
    "base_gen_AQI/sulfur dioxide_forecast_2025-12-10_11-15-34.csv",
    "base_gen_AQI/nitrogen dioxide (no2)_forecast_2025-12-10_10-54-45.csv",
    "base_gen_AQI/pm2.5 - local conditions_forecast_2025-12-10_11-12-07.csv",
    "base_gen_AQI/pm10 total 0-10um stp_forecast_2025-12-10_10-58-28.csv"
}

model = train_joint_vae(filepaths)
samples = generate_year_of_aqi(model)
df = flatten_to_hourly_csv_with_total(samples)
df.to_csv("synthetic_aqi_year.csv", index=False)