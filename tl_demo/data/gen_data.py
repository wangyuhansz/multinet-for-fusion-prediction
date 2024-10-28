import os
import numpy as np


def gen_data(N, a_range, x_range, dist, fidelity):
    """Generates data points for DEMO.

    Args:
        N (int): Number of data points to generate.
        a_range (list): Range of 'a' values as [min, max].
        x_range (list): Range of 'x' values as [min, max].
        dist (str): Distribution type ('random' or 'uniform').
        fidelity (str): Fidelity level ('Exp', 'High', 'Low').
    """
    if dist == 'random':
        a_values = np.random.uniform(a_range[0], a_range[1], N)
        x_values = np.random.uniform(x_range[0], x_range[1], N)
    elif dist == 'uniform':
        a_values = np.linspace(a_range[0], a_range[1], N)
        x_values = np.linspace(x_range[0], x_range[1], N)

    if fidelity == 'exp':
        f_values = x_values * np.exp(a_values * x_values.astype(float))
    elif fidelity == 'high':
        f_values = x_values + a_values * x_values**2
    elif fidelity == 'low':
        f_values = x_values

    data = np.column_stack((a_values, x_values, f_values))

    output_dir = os.path.join(fidelity, dist)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'data.npy'), data)


def main():
    data_distribution = ['random', 'uniform']
    data_fidelity = ['exp', 'high', 'low']

    N = 1000
    a_range = [0, 1]
    x_range = [-1, 1]

    for dist in data_distribution:
        for fidelity in data_fidelity:
            gen_data(N, a_range, x_range, dist, fidelity)


if __name__ == '__main__':
    main()
