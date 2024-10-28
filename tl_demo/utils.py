import os
import yaml
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class MYDATA(Dataset):
    def __init__(self, data):
        self.normalized_data, self.min_vals, self.max_vals = self.normalize(data)

    def __len__(self):
        return self.normalized_data.size(0)

    def __getitem__(self, idx):
        Prob = self.normalized_data[idx, 0:2]
        label = self.normalized_data[idx, 2]
        return Prob, label

    def normalize(self, data):
        min_vals = data.min(0)
        max_vals = data.max(0)

        normalized_data = (data - min_vals) / (max_vals - min_vals)
        normalized_data = torch.tensor(normalized_data, dtype=torch.float32)
        return normalized_data, min_vals, max_vals

    def inverse_normalize(self, normalized_data): 
        return normalized_data * (self.max_vals - self.min_vals) + self.min_vals


def get_data(args):
    """ Gets the data for training or transfer learning.
    """
    def data_loader(dist, fidelity, tl=False):
        data = np.load(rf".\data\{fidelity}\{dist}\data.npy")
        data = data[:int(0.05 * data.shape[0]), :] if tl else data # For TL, take only the first 5% of the data
        dataset = MYDATA(data)
        data_range = [dataset.min_vals[-1], dataset.max_vals[-1]]
        
        # Split the dataset into training and validation sets (80:20 ratio)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        if tl:
            train_data = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
            val_data = DataLoader(val_dataset, batch_size=len(val_dataset))
        else:
            train_data = DataLoader(train_dataset, batch_size=64, drop_last=True, shuffle=True)
            val_data = DataLoader(val_dataset, batch_size=64, drop_last=True)
        
        return train_data, val_data, data_range

    if args.mode == 'train':
        if args.fidelity == 'low' or 'high' or 'exp':
            return data_loader(args.dist, args.fidelity)
        else:
            raise ValueError("Invalid fidelity for train mode. Please choose either 'low', 'high', or 'exp'.")
    elif args.mode == 'tl':
        if args.fidelity == 'low2high' or 'low2exp' or 'high2exp' or 'low2high2exp':
            _, higher_fidelity = get_tl_fidelity(args.fidelity)
            return data_loader(args.dist, higher_fidelity, tl=True)
        else:
            raise ValueError("Invalid fidelity for tl mode. Please choose either 'low2high', 'low2exp', 'high2exp', or 'low2high2exp'.")
    else:
        raise ValueError("Invalid mode. Please choose either 'train' or 'tl'.")


def inverse_normalize(normalized_data, data_range): 
    [min_vals, max_vals] = data_range
    return normalized_data * (max_vals - min_vals) + min_vals


def prepare_args():
    """ Prepares the arguments for the training process from a YAML configuration file.
    
    Returns:
        argparse.Namespace: An object containing the configuration parameters.
    """
    def get_mode(fidelity):
        if fidelity in ['low', 'high', 'exp']:
            return 'train'
        elif fidelity in ['low2high', 'low2exp', 'high2exp', 'low2high2exp']:
            return 'tl'
        else:
            raise ValueError("Invalid fidelity.")
    
    parser = argparse.ArgumentParser(description='DEMO for Transfer Learning for ICF')
    
    parser.add_argument('--config', type=str, default='./configs/mlp.yaml', help='Configuration file to use')
    with open(parser.parse_known_args()[0].config, 'r') as file:
        config = yaml.safe_load(file)

    parser.add_argument('--dist', type=str, default=config.get('dist', 'random'), 
                        help='Distribution type of the data, random or uniform')
    parser.add_argument('--fidelity', type=str, default=config.get('fidelity', 'low'), 
                        help='Fidelity level of the data, low, high, exp, low2high, low2exp, high2exp or low2high2exp')
    parser.add_argument('--model', type=str, default=config.get('model', 'MLP'), 
                        help='Model type to use for training, MLP, CNN, RNN or LSTM')
    parser.add_argument('--dim_layers', type=int, nargs='+', default=config.get('dim_layers', [2, 10, 10, 10, 5, 1]), 
                        help='Dimensions of the layers in the model')
    parser.add_argument('--activation', type=str, default=config.get('activation', 'PReLU'), 
                        help='Activation function to use in the model, PReLU or ReLU')
    parser.add_argument('--optimizer', type=str, default=config.get('optimizer', 'Adam'), 
                        help='Optimizer to use: Adam, SGD')
    parser.add_argument('--loss_fn', type=str, default=config.get('loss_fn', 'MSE'), 
                        help='Loss function to use: MSE')
    parser.add_argument('--epochs', type=int, default=config.get('epochs', 1001), 
                        help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 64), 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.get('lr', 1e-3), 
                        help='Learning rate for the optimizer')
    parser.add_argument('--val_interval', type=int, default=config.get('val_interval', 1), 
                        help='Validation interval in terms of epochs')
    parser.add_argument('--seed', type=int, default=config.get('seed', 42), 
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=config.get('device', 'default'), 
                        help='Device to use for training, cuda or cpu, default for choosing automatically')

    parser.add_argument('--mode', type=str, default=get_mode(parser.parse_known_args()[0].fidelity), 
                        help='Mode to run the script: train or tl')

    args = parser.parse_args()
    return args


def get_device(device):
    if device == 'default':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device == 'cuda' or 'cpu':
        device = torch.device(device)
    else:
        raise ValueError("Invalid device. Please choose either 'cuda' or 'cpu', or 'default' for automatic selection.")
    return device


def get_optimizer(optimizer, model, lr):
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer. Please choose either 'Adam' or 'SGD'.")
    return optimizer


def get_loss_fn(loss_fn):
    if loss_fn == 'MSE':
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError("Invalid loss function. Please choose 'MSE'.")
    return loss_fn


def get_model_name(args):
    model_name = "{}_{}_{}_{}_{}_{}_{}".format(
        args.model, 
        args.activation, 
        args.optimizer, 
        args.loss_fn,
        args.epochs,
        args.batch_size,
        "{:.0e}".format(args.lr) #* KEEP ONLY ONE significant digit for lr
        )
    
    if args.model == 'MLP':
        model_name += f"_{'_'.join(map(str, args.dim_layers))}"
    
    return model_name


def get_model_dir(args, mode='models', fidelity=None, model_name=None):
    fidelity = args.fidelity if fidelity is None else fidelity
    model_name = get_model_name(args) if model_name is None else model_name

    model_path = "./{}/{}/{}/{}/".format(
        mode,
        fidelity, 
        args.dist, 
        model_name
        )
    
    return model_path


def get_tl_load_path(args):
    """ Find the latest .pth file based on the timestamp in the loading directory"""
    lower_fidelity, _ = get_tl_fidelity(args.fidelity)
    load_dir = get_model_dir(args, fidelity=lower_fidelity)
    if os.path.exists(load_dir):
        pth_files = [f for f in os.listdir(load_dir) if f.endswith('.pth')]
        if pth_files:
            latest_file = max(pth_files, key=lambda x: datetime.strptime(x.split('.')[0], "%Y%m%d_%H%M%S"))
            load_path = os.path.join(load_dir, latest_file)
        else:
            raise FileNotFoundError("No .pth files found in the directory.")
    else:
        raise FileNotFoundError(f"The directory {load_path} does not exist.")
    print(f"Loading model from {load_path}")
    return load_path


def get_save_log_plot_path(args):
    """ Returns the save path and log directory for the model. 
        The two paths are based on the same timestamp.
    """
    save_dir = get_model_dir(args, mode='models')
    log_dir = get_model_dir(args, mode='logs')
    plot_dir = get_model_dir(args, mode='plots')

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(save_dir, time_stamp + ".pth")
    log_dir = os.path.join(log_dir, time_stamp)
    plot_dir = os.path.join(plot_dir, time_stamp + "/")

    return save_path, log_dir, plot_dir


def get_tl_fidelity(fidelity):
    """ Returns the tuple of fidelity levels (source, target) for transfer learning.
    """
    if fidelity == 'low2high':
        tl_fidelity = ('low', 'high')
    elif fidelity == 'low2exp':
        tl_fidelity = ('low', 'exp')
    elif fidelity == 'high2exp':
        tl_fidelity = ('high', 'exp')
    elif fidelity == 'low2high2exp':
        tl_fidelity = ('low2high', 'exp')
    else:
        raise ValueError("Invalid fidelity of tl. Please choose either 'low2high', 'low2exp', 'high2exp' or 'low2high2exp'.")
    return tl_fidelity
