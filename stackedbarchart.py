import pandas as pd
import matplotlib.pyplot as plt

def plot_stacked_forecasts(filepaths):

    pollutant_data = {}

    for pollutant, path in filepaths.items():
        df = pd.read_csv(path)

        # Assume each forecast CSV has an 'AQI' column with 24 hourly predictions
        pollutant_data[pollutant] = df['AQI'].values[:24]

    # Combine into single DataFrame
    combined = pd.DataFrame(pollutant_data)
    combined.index = range(24)  # hours 0–23

    # Plot stacked bar chart
    ax = combined.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 6),
        colormap='tab20'
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted AQI Contribution")
    ax.set_title("Stacked AQI Forecasts by Pollutant (24 Hours)")
    plt.legend(title="Pollutant")
    plt.tight_layout()
    plt.show()


filepaths = {
    "CO": "base_gen_AQI/carbon monoxide_forecast_2025-12-10_10-26-59.csv",
    "O3": "base_gen_AQI/ozone_forecast_2025-12-09_23-55-08.csv",
    "SO2": "base_gen_AQI/sulfur dioxide_forecast_2025-12-10_11-15-34.csv",
    "NO2": "base_gen_AQI/nitrogen dioxide (no2)_forecast_2025-12-10_10-54-45.csv",
    "PM2.5": "base_gen_AQI/pm2.5 - local conditions_forecast_2025-12-10_11-12-07.csv",
    "PM10": "base_gen_AQI/pm10 total 0-10um stp_forecast_2025-12-10_10-58-28.csv"
}

plot_stacked_forecasts(filepaths)