import os
import yaml
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class NormalizedData(Dataset):
    def __init__(self, data_raw, val_data_raw):
        self.normalized_data, self.min_vals, self.max_vals = self.normalize(data_raw, val_data_raw)

    def __len__(self):
        return self.normalized_data.size(0)

    def __getitem__(self, idx):
        Prob = self.normalized_data[idx, :-1]
        label = self.normalized_data[idx, -1]
        return Prob, label

    def normalize(self, data_raw, val_data_raw):
        data = np.concatenate((data_raw, val_data_raw), axis=0)
        min_vals = data.min(0)
        max_vals = data.max(0)  # + 1e-6  # To avoid division by zero

        normalized_data = (data - min_vals) / (max_vals - min_vals)
        normalized_data = torch.tensor(normalized_data, dtype=torch.float32)  # Convert to tensor
        return normalized_data, min_vals, max_vals


def get_data(args):
    """ Gets the data for training or transfer learning. """
    # If the mode is transfer learning, the fidelity level is the higher one
    fidelity = get_tl_fidelity(args.fidelity)[1] if args.mode == 'tl' else args.fidelity
    
    if args.dist == 'default':   # _dist_ribution is an abandoned parameter
        data_raw = np.load(rf"./data/{fidelity}/{args.data_name}.npy")
        val_data_raw = np.load(rf"./data/{fidelity}/{args.val_data_name}.npy")
    else:
        data_raw = np.load(rf"./data/{fidelity}/{args.dist}/{args.data_name}.npy")
        val_data_raw = np.load(rf"./data/{fidelity}/{args.dist}/{args.val_data_name}.npy")

    if args.target == 'V_mean':
        target_index = -7
    elif args.target == 'rhor_DT':
        target_index = -9
    elif args.target == 'timestag':
        target_index = -3
    elif args.target == 'rho_mean':
        target_index = -4
    else:
        raise ValueError("Invalid target variable. Please choose either 'V_mean' or 'rhor_DT'.")
    
    data_raw = data_raw[:, [13] + list(range(16, 22)) + [target_index]]
    val_data_raw = val_data_raw[:, [13] + list(range(16, 22)) + [target_index]]

    data_size = data_raw.shape[0]

    concat_dataset = NormalizedData(data_raw, val_data_raw)
    data_range = [concat_dataset.min_vals[-1], concat_dataset.max_vals[-1]]

    dataset = torch.utils.data.Subset(concat_dataset, range(data_size))
    val_dataset = torch.utils.data.Subset(concat_dataset, range(data_size, len(concat_dataset)))
    
    # Split the dataset into training and validation sets (80:20 ratio)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f'Training data size: {len(train_dataset)}, Test data size: {len(test_dataset)}, Validation data size: {len(val_dataset)}')
    
    drop_last = False if args.mode == 'tl' else True
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=drop_last, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=int(args.batch_size/4), drop_last=drop_last)
    val_data = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=drop_last)
    
    return train_data, test_data, val_data, data_range


def inverse_normalize(normalized_data, data_range): 
    [min_vals, max_vals] = data_range
    return normalized_data * (max_vals - min_vals) + min_vals


def get_data_range(args, target):
    """ Directly get the data range. """
    data_raw = np.load(rf"./data/{args.fidelity}/{args.data_name}.npy")
    val_data_raw = np.load(rf"./data/{args.fidelity}/{args.val_data_name}.npy")

    if target == 'V_mean':
        target_index = -7
    elif target == 'rhor_DT':
        target_index = -9
    elif target == 'timestag':
        target_index = -3
    elif target == 'rho_mean':
        target_index = -4
    
    data_raw = data_raw[:, [13] + list(range(16, 22)) + [target_index]]
    val_data_raw = val_data_raw[:, [13] + list(range(16, 22)) + [target_index]]
    
    data = np.concatenate((data_raw, val_data_raw), axis=0)
    min_vals = data.min(0)
    max_vals = data.max(0)
    return min_vals, max_vals


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
    
    def print_args(args):
        """ Prints the arguments in a readable format. """
        print("Arguments:")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

    parser = argparse.ArgumentParser(description='DEMO for Transfer Learning for ICF')
    
    parser.add_argument('--config', type=str, default='./configs/mlp.yaml', help='Configuration file to use')
    with open(parser.parse_known_args()[0].config, 'r') as file:
        config = yaml.safe_load(file)

    parser.add_argument('--dist', type=str, default=config.get('dist', 'default'), 
                        help='Distribution type of the data, random or uniform')
    parser.add_argument('--fidelity', type=str, default=config.get('fidelity', 'low'), 
                        help='Fidelity level of the data, low, high, exp, low2high, low2exp, high2exp or low2high2exp')
    parser.add_argument('--model', type=str, default=config.get('model', 'MLP'), 
                        help='Model type to use for training, MLP, CNN, RNN or LSTM')
    parser.add_argument('--dim_layers', type=int, nargs='+', default=config.get('dim_layers', [22, 22, 22, 10, 1]), # [2, 10, 10, 10, 5, 1]
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
    parser.add_argument('--mode', type=str, default=config.get('mode', 'test'),  # get_mode(parser.parse_known_args()[0].fidelity), 
                        help='Mode to run the script: train, tl, test or val') #!
    parser.add_argument('--data_name', type=str, default=config.get('data_name', 'data'),
                        help='Name of the data file to use')
    parser.add_argument('--val_data_name', type=str, default=config.get('val_data_name', 'val_data'),
                        help='Name of the validation data file to use')
    parser.add_argument('--target', type=str, default=config.get('target', 'V_mean'),
                        help='Target variable to predict, V_mean or rhor_DT')
    
    args = parser.parse_args()
    print_args(args)
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


def get_model_dir(args, mode='models', fidelity=None, model_name=None, target=None):
    fidelity = args.fidelity if fidelity is None else fidelity
    model_name = get_model_name(args) if model_name is None else model_name
    data_name = args.val_data_name if mode == 'plots' and args.mode == 'val' else args.data_name
    target = args.target if target is None else target
    
    if args.dist == 'default':
        model_path = "./{}/{}/{}/{}/{}/".format(
            mode,
            fidelity,
            target,
            data_name,
            model_name
            )
    else:
        model_path = "./{}/{}/{}/{}/{}/{}/".format(
            mode,
            fidelity, 
            target,
            args.dist,
            data_name,
            model_name
            )
    
    return model_path


def get_latest_load_path(args, fidelity=None, target=None):
    """ Find the latest model folder based on the timestamp in the loading directory"""
    fidelity = args.fidelity if fidelity is None else fidelity
    target = args.target if target is None else target
    load_dir = get_model_dir(args, fidelity=fidelity, target=target)

    if os.path.exists(load_dir):
        folders = [f for f in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, f))]
        if folders:
            latest_folder = max(folders)
            load_path = os.path.join(load_dir, latest_folder)
        else:
            raise FileNotFoundError("No .pth files found in the directory.")
    else:
        raise FileNotFoundError(f"The directory {load_dir} does not exist.")
    
    print(f"Loading model from {load_path}")
    return load_path


def get_best_model_path(args, load_dir=None, target=None):
    """ Find the best model path based on the specific criterion. """

    def criterion(training_loss, test_loss, val_loss):
        """ Returns the weighted sum of the losses. """
        return training_loss + test_loss + 0.02 * val_loss  #! Change the weight here to an adjustable argument
    
    target = args.target if target is None else target
    load_dir = get_latest_load_path(args, target=target) if load_dir is None else load_dir
    load_dir = load_dir if os.path.exists(load_dir) else get_latest_load_path(args, target=target)
    train_stats = np.genfromtxt(os.path.join(load_dir, 'train_stats.csv'), delimiter=',', skip_header=1)
    
    best_epoch = None
    best_criterion_value = float('inf')
    
    for row in train_stats:
        epoch, training_loss, test_loss, val_loss = row
        criterion_value = criterion(training_loss, test_loss, val_loss)
        
        if criterion_value < best_criterion_value:
            best_criterion_value = criterion_value
            best_epoch = int(epoch)
    
    if best_epoch is None:
        raise ValueError("No valid epoch found in train_stats.csv.")
    
    best_model_path = os.path.join(load_dir, f"{best_epoch}.pth")
    print(f"Best model found at epoch {best_epoch} with criterion value {best_criterion_value}")
    
    return best_model_path


def get_tl_load_path(args):
    """ Find the latest .pth file based on the timestamp in the loading directory"""
    lower_fidelity, _ = get_tl_fidelity(args.fidelity)
    load_path = get_latest_load_path(args, fidelity=lower_fidelity)
    best_model_path = get_best_model_path(args, load_path)
    return best_model_path


def get_save_log_plot_path(args):
    """ Returns the save path and log directory for the model. 
        The two paths are based on the same timestamp.
    """
    save_dir = get_model_dir(args, mode='models')
    log_dir = get_model_dir(args, mode='logs')
    plot_dir = get_model_dir(args, mode='plots')

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(save_dir, time_stamp + "/")
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
