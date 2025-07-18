import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def exp_decay(t, A, tau, offset):
    """Single exponential decay model."""
    return A * np.exp(-t / tau) + offset


def main():
    # Load Excel file with columns 'time' and 'voltage'
    data = pd.read_excel('sample_data.xlsx')
    t = data['time'].values
    v = data['voltage'].values

    # Initial guesses for parameters: amplitude, time constant, offset
    guess = (v.max() - v.min(), (t.max() - t.min()) / 2.0, v.min())

    popt, pcov = curve_fit(exp_decay, t, v, p0=guess)

    # Generate a smooth curve for plotting
    t_fit = np.linspace(t.min(), t.max(), 500)
    v_fit = exp_decay(t_fit, *popt)

    # Calculate residuals
    residuals = v - exp_decay(t, *popt)

    # Plot the data, the fitted curve and residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    ax1.scatter(t, v, label='Data')
    ax1.plot(t_fit, v_fit, color='red', label='Fit')
    ax1.set_ylabel('Voltage')
    ax1.legend()

    ax2.scatter(t, residuals, color='purple', label='Residuals')
    ax2.axhline(0, color='grey', linestyle='--')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Residual')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
