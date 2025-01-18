import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import utils, networks


def main():
    args = utils.prepare_args()
    target_x, target_y = 'rho_mean', 'V_mean' # 'rhor_DT', 'timestag', 'V_mean'

    # Initialize the models, using the same architecture 
    model_x = networks.get_model(args)
    model_y = networks.get_model(args)

    plot_path = utils.get_save_log_plot_path(args)[2]

    device = utils.get_device(args.device)
    model_x.to(device)
    model_y.to(device)

    model_x.load_state_dict(torch.load(utils.get_best_model_path(args, target=target_x)))
    model_y.load_state_dict(torch.load(utils.get_best_model_path(args, target=target_y))) 

    x_min_vals, x_max_vals = utils.get_data_range(args, target_x)
    y_min_vals, y_max_vals = utils.get_data_range(args, target_y) 
    min_vals = x_min_vals[:-1]
    max_vals = x_max_vals[:-1]

    benchmark_means = np.array([4.346, 3.393, 3.9423, 4.831, 5.633, 12.174, 22.4]) #! Change for the calibrated
    benchmark_means = (benchmark_means - min_vals) / (max_vals - min_vals)
    benchmark_stds = benchmark_means * 0.25 # 0.7

    data = torch.normal(mean=torch.tensor(benchmark_means, dtype=torch.float32).unsqueeze(0).expand(N, -1), 
                        std=torch.tensor(benchmark_stds, dtype=torch.float32).unsqueeze(0).expand(N, -1))
    
    # data = np.load(rf"./data/{args.fidelity}/{args.data_name}.npy")[:100, [13] + list(range(16, 22)) + [-7]]
    # V_mean_labels = data[:, -1]
    # data = (data[:, :-1] - min_vals) / (max_vals - min_vals)

    # rhor_DT_labels = np.load(rf"./data/{args.fidelity}/{args.val_data_name}.npy")[:100, -7]

    # print(f"data:{data}")
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    data_tensor = data.to(device)
    with torch.no_grad():
        x_predictions = model_x(data_tensor).cpu().numpy()
        y_predictions = model_y(data_tensor).cpu().numpy()

    x_predictions = x_predictions * (x_max_vals[-1] - x_min_vals[-1]) + x_min_vals[-1]
    y_predictions = y_predictions * (y_max_vals[-1] - y_min_vals[-1]) + y_min_vals[-1]
    # print(f"x_predictions first 5: {x_predictions[:5, :]}")
    # print(f"y_predictions first 5: {y_predictions[:5, :]}")

    plt.figure(figsize=(10, 8), facecolor='none')
    plt.hist2d(y_predictions.flatten(), x_predictions.flatten(), bins=400, 
               cmap=LinearSegmentedColormap.from_list('my',[(1,1,1), (0.5,0,0)], 10))
    
    plt.xlim([153, 163])
    plt.ylim([145, 195])

    plt.axhline(y=180, color=(0.75,0,0), linestyle='--', linewidth=2)
    plt.text(153.2, 180.8, '28kJ', fontsize=12)

    plt.colorbar(label='Density')
    plt.xlabel(f'{target_y}_predictions')
    plt.ylabel(f'{target_x}_predictions')
    plt.title('Density Heatmap')

    #! If you want to save the plot, uncomment the following 4 lines
    # os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    # plot_save_path = os.path.join(os.path.dirname(plot_path), 'density_heatmap.png')
    # plt.savefig(plot_save_path, dpi=800)
    # print(f"Plot saved at {plot_save_path}")

    plt.show()


if __name__ == "__main__":
    N = int(2e7)

    main()
