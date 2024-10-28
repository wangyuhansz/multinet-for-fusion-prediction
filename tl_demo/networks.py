from collections import OrderedDict
from torch import nn
from torch.nn import PReLU, ReLU


class MLP(nn.Module):
    def __init__(self, dim_layers, activation='PReLU'):
        super(MLP, self).__init__()
        
        layers = []
        for i in range(len(dim_layers) - 1):
            layers.append((f'layer_{i}', nn.Linear(dim_layers[i], dim_layers[i + 1])))

            # No activation after the last layer
            if i < len(dim_layers) - 2:
                layers.append((f'activation_{i}', get_activation_func(activation)))
        
        self.mlp = nn.Sequential(OrderedDict(layers))

    def forward(self, input):
        output = self.mlp(input)
        return output


#! TODO: The following classes are not carefully tested yet!
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='PReLU'):
        super(CNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            get_activation_func(activation)
        )

    def forward(self, input):
        output = self.cnn(input)
        return output


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation='PReLU'):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.activation = get_activation_func(activation)
    
    def forward(self, input):
        output, _ = self.rnn(input)
        output = self.activation(output)
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation='PReLU'):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.activation = get_activation_func(activation)
    
    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.activation(output)
        return output


def get_activation_func(activation):
    if activation == 'PReLU':
        activation_func = PReLU()
    elif activation == 'ReLU':
        activation_func = ReLU()
    else:
        raise ValueError("Invalid activation function. Please choose either 'PReLU' or 'ReLU'.")
    return activation_func


def get_model(args):
    if args.model == 'MLP':
        model = MLP(args.dim_layers, args.activation)
    elif args.model == 'CNN':
        model = CNN(args.in_channels, args.out_channels, args.kernel_size, 
                             args.stride, args.padding, args.activation)
    elif args.model == 'RNN':
        model = RNN(args.input_size, args.hidden_size, args.num_layers, args.activation)
    elif args.model == 'LSTM':
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.activation)
    else:
        raise ValueError("Invalid model. Please choose either 'MLP', 'CNN', 'RNN', or 'LSTM'.")
    return model
