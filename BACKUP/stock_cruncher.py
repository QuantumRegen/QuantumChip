import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


# Generate a manageable dataset (stock prices)
def generate_stock_data(n_rows=500_000):  # Halved from 1M to fit current RAM comfy
    print(f"Generating {n_rows:,} rows of fake stock data...")
    dates = pd.date_range("2020-01-01", periods=n_rows // 100, freq="T")
    data = pd.DataFrame({
        "Date": np.tile(dates, 100),
        "Price": np.random.uniform(50, 150, n_rows),
        "Volume": np.random.randint(100, 10000, n_rows)
    })
    return data


# Crunch numbers in parallel
def analyze_chunk(chunk):
    return {
        "Mean_Price": chunk["Price"].mean(),
        "Max_Volume": chunk["Volume"].max(),
        "Volatility": chunk["Price"].std()
    }


def parallel_analysis(data, num_processes=16):  # Match your 16 cores
    print(f"Crunching with {num_processes} cores...")
    chunks = np.array_split(data, num_processes)
    with Pool(num_processes) as p:
        results = p.map(analyze_chunk, chunks)
    return results


# Plot it
def plot_results(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data["Date"][::500], data["Price"][::500], label="Stock Price")
    plt.title("Stock Price Over Time (Sampled)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# Main
if __name__ == "__main__":
    start_time = time.time()

    # Generate data
    stock_data = generate_stock_data()
    print(f"Data size: {stock_data.memory_usage().sum() / 1024 ** 2:.2f} MB")

    # Parallel crunch
    results = parallel_analysis(stock_data)
    mean_price = np.mean([r["Mean_Price"] for r in results])
    max_volume = max([r["Max_Volume"] for r in results])
    volatility = np.mean([r["Volatility"] for r in results])

    print(f"Mean Price: ${mean_price:.2f}")
    print(f"Max Volume: {max_volume:,}")
    print(f"Volatility: ${volatility:.2f}")

    # Plot
    plot_results(stock_data)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")