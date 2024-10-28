import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import networks
from utils import inverse_normalize


def plot_pre_exp_results(args, val_data, data_range, device, save_path, plot_path):
    model = networks.get_model(args)
    model.load_state_dict(torch.load(save_path))
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

    real_predictions = inverse_normalize(predictions, data_range)
    real_actuals = inverse_normalize(actuals, data_range)
    
    mean_squared_error = np.mean((real_predictions - real_actuals) ** 2)

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plot_path += "predictions_exp_results.png"

    plt.figure()
    plt.scatter(real_predictions, real_actuals)
    plt.plot(data_range, data_range, linewidth=1, linestyle='--', color='r')
    plt.xlabel("Predictions")
    plt.ylabel("Experimental Results")
    plt.title("Predictions vs Experimental Results")
    plt.legend(["Mean Squared Error: {:.2e}".format(mean_squared_error)])
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved at {plot_path}")
