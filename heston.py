import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd


def calculate_vol_of_vol(price_data, window=30):
    """
    Calculate the volatility of the volatility

    Parameters
    ----------
    price_data : np.ndarray
        The observed price data for the asset
    window : int
        The window for the rolling volatility

    Returns
    -------
    xi : float
        The volatility of the volatility
    """
    # Calculate rolling volatility using square root of time rule
    volatility = np.array(pd.Series(price_data).rolling(
        window).std()) * 1/np.sqrt(window)
    # Make the volatility data the same length as the price data
    volatility = volatility[~np.isnan(volatility)]
    # Calculate the volatility of the volatility using square root of time rule
    xi = np.std(volatility) * 1/np.sqrt(len(volatility))
    return xi


def calculate_correlation_between_asset_price_and_volatility(price_data, window=30):
    """
    Calculate the correlation between the asset price and its volatility

    Parameters
    ----------
    price_data : np.ndarray
        The observed price data for the asset

    Returns
    -------
    rho : float
        The correlation between the asset price and its volatility
    """
    # Calculate rolling volatility

    volatility = np.array(pd.Series(price_data).rolling(window).std())
    # Make the volatility data the same length as the price data
    volatility = volatility[~np.isnan(volatility)]
    # Calculate the correlation between the asset price and its volatility
    rho = np.corrcoef(price_data[window-1:], volatility)[0, 1]
    return rho


def heston_model(S0, T, r, kappa, theta, xi, rho, N):
    """
    Simulate the price of a stock using the Heston model

    Parameters
    ----------
    S0 : float
        The initial price of the asset
    T : float
        The time to maturity
    r : float
        The risk-free interest rate
    kappa : float
        The mean-reversion rate 
    theta : float
        The long-run average volatility
    xi : float
        The volatility of the volatility
    rho : float
        The correlation between the asset price and its volatility
    N : int
        The number of time steps


    Returns
    -------
    price_path : np.ndarray
        The simulated price path of the stock using the Heston model
    """
    # Set the initial price
    price_path = np.zeros(N)
    # Set the time step
    dt = T / N
    # Simulate the price path

    # Set the initial price
    price_path[0] = S0
    # Set the initial volatility
    v = theta
    # Simulate the price path
    for t in range(1, N):
        # Generate the random shocks
        z1 = np.random.normal(0, 1)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1)
        # Calculate the new volatility
        v = v + kappa * (theta - v) * dt + xi * np.sqrt(v * dt) * z2
        # Calculate the new price
        price_path[t] = price_path[t - 1] * \
            np.exp((r - 0.5 * v) * dt + np.sqrt(v * dt) * z1)
    return price_path


def plot_price_path(price_path):
    """
    Plot the price path
    """
    plt.plot(price_path)
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.title('Simulated Asset Price Path')
    plt.show()


def plot_multiple_price_paths(price_paths):
    """
    Plot the price path
    """
    for price_path in price_paths:
        plt.plot(price_path)
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.title('Simulated Asset Price Path')
    plt.show()


if __name__ == "__main__":
    # Create some observed data
    # Define the parameters
    S0 = 3250
    T = 1
    r = 0.05
    kappa = 0.2
    theta = 2.5
    xi = 0.2
    rho = 0.9
    N = 365 * 10
    price_paths = [heston_model(S0, T, r, kappa, theta, xi, rho, N)
                   for _ in range(10)]
    plot_multiple_price_paths(price_paths)
