import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import networks
from utils import inverse_normalize, get_latest_load_path, get_best_model_path


def plot_pre_exp_results(args, val_data, data_range, device, save_path, plot_path, mode='test'):
    model = networks.get_model(args)
    save_path = save_path if os.path.exists(save_path) else get_latest_load_path(args)
    best_model_path = get_best_model_path(args, save_path)
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        all_pre_results = []
        all_exp_results = []
        
        for data, target in val_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pre_results = output.squeeze(1).cpu().numpy()
            all_pre_results.extend(pre_results)
            all_exp_results.extend(target.cpu().numpy())
        
        predictions = np.array(all_pre_results)
        actuals = np.array(all_exp_results)

    mean_squared_error = np.mean((predictions - actuals) ** 2)

    real_predictions = inverse_normalize(predictions, data_range)
    real_actuals = inverse_normalize(actuals, data_range)

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    if mode == 'val':
        plot_path += "predictions_val_results.png"
    elif mode == 'test':
        plot_path += "predictions_test_results.png"

    plt.figure()
    plt.scatter(real_predictions, real_actuals, s=2)
    plt.plot(data_range, data_range, linewidth=1, linestyle='--', color='r')
    plt.xlabel("Predictions")
    if mode == 'val':
        plt.ylabel("Validation Results")
        plt.title("Predictions vs Validation Results")
    elif mode == 'test':
        plt.ylabel("Test Results")
        plt.title("Predictions vs Test Results")

    # plt.xlim([165, 205])
    # plt.ylim([165, 205])
    
    plt.legend([f"{args.target} MSE: {mean_squared_error:.4e}"])
    plt.tight_layout()
    plt.savefig(plot_path, dpi=800, transparent=True)
    print(f"Plot saved at {plot_path}")
