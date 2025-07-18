import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def exp_decay(t, A, tau, offset):
    """Single exponential decay model."""
    return A * np.exp(-t / tau) + offset


def main():
    # Load CSV with columns 'time' and 'voltage'
    data = pd.read_csv('cs_data.csv')
    t = data['time'].values
    v = data['voltage'].values

    # Initial guesses for parameters: amplitude, time constant, offset
    guess = (v.max() - v.min(), (t.max() - t.min()) / 2.0, v.min())

    popt, pcov = curve_fit(exp_decay, t, v, p0=guess)

    # Generate a smooth curve for plotting
    t_fit = np.linspace(t.min(), t.max(), 500)
    v_fit = exp_decay(t_fit, *popt)

    # Plot the data and the fitted curve
    plt.figure()
    plt.scatter(t, v, label='Data')
    plt.plot(t_fit, v_fit, color='red', label='Fit')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
