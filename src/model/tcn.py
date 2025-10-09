import torch.nn as nn
import torch


class Chomp1d(nn.Module):
    """Removes the extra padding from the end of a sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, number_of_layers, kernel_size, stride, dilation,
                 dropout=0.2, norm='weight_norm', activation='ReLU'):
        super(TemporalBlock, self).__init__()

        layers = []
        in_channels = n_inputs
        for i in range(number_of_layers):
            # Calculate padding for causality
            padding = (kernel_size - 1) * dilation

            # Define the convolutional layer
            conv_layer = nn.Conv1d(in_channels, n_outputs, kernel_size, stride=stride,
                                   padding=padding, dilation=dilation)

            # Apply weight normalization if specified
            if norm == 'weight_norm':
                conv_layer = nn.utils.parametrizations.weight_norm(conv_layer)

            layers.append(conv_layer)

            # Add normalization layer if it's not weight_norm
            if norm and norm != 'weight_norm':
                layers.append(getattr(nn, norm)(n_outputs))

            # Add Chomp1d to remove padding and ensure causality
            layers.append(Chomp1d(padding))
            # Add flexible activation function
            layers.append(getattr(nn, activation)())
            # Add dropout
            layers.append(nn.Dropout(dropout))

            # The input channels for the next layer will be the output channels of this one
            in_channels = n_outputs

        self.network = nn.Sequential(*layers)

        # Downsample layer for the residual connection if channel numbers differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # Final activation for the residual connection
        self.relu = getattr(nn, activation)()
        self.init_weights()

    def init_weights(self):
        # Initialize weights for all convolutional layers in the network
        for m in self.network:
            if isinstance(m, nn.Conv1d):
                # weight_norm re-parameterizes the layer, so we initialize the 'weight_v' parameter
                if hasattr(m, 'weight_v'):
                    # Use normal distribution with smaller std for more stable initialization
                    nn.init.normal_(m.weight_v, mean=0.0, std=0.01)
                    # Ensure weight_g is initialized properly
                    if hasattr(m, 'weight_g'):
                        nn.init.constant_(m.weight_g, 1.0)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                
                # Initialize bias to zero
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, mean=0.0, std=0.01)
            if self.downsample.bias is not None:
                nn.init.constant_(self.downsample.bias, 0.0)

    def forward(self, x):
        out = self.network(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, number_of_layers_per_block, kernel_size=2,
                 dropout=0.2, dilations=None, norm='weight_norm', activation='ReLU'):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_blocks = len(num_channels)
        
        # Default dilations if not provided: [1, 2, 4, 8, ...]
        if dilations is None:
            dilations = [2**i for i in range(num_blocks)]

        for i in range(num_blocks):
            dilation_size = dilations[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, number_of_layers_per_block,
                                        kernel_size, stride=1, dilation=dilation_size,
                                        dropout=dropout, norm=norm, activation=activation))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    """TCN model for joint moment prediction from IMU data."""
    def __init__(self, hyperparameter_config):
        super(TCNModel, self).__init__()
        
        # Configuration parameters
        self.input_size = hyperparameter_config['input_size']  # Number of sensor inputs
        self.output_size = hyperparameter_config['output_size']  # Number of joint moments
        self.num_channels = hyperparameter_config['num_channels']  # Channels per TemporalBlock
        self.kernel_size = hyperparameter_config['kernel_size']  # Kernel size for convolutional layers
        self.number_of_layers = hyperparameter_config['number_of_layers']
        self.dropout = hyperparameter_config['dropout']  # Dropout value
        self.dilations = hyperparameter_config['dilations']  # Dilations for TemporalBlocks
        self.window_size = hyperparameter_config['window_size']  # Get the sequence length
        self.norm = hyperparameter_config.get('norm', 'weight_norm')  # Normalization type (default: weight_norm)
        self.activation = hyperparameter_config.get('activation', 'ReLU')  # Activation function (default: ReLU)
        
        self.tcn = TemporalConvNet(self.input_size, self.num_channels, self.number_of_layers, 
                                   self.kernel_size, self.dropout, self.dilations, 
                                   norm=self.norm, activation=self.activation)
        self.linear = nn.Linear(self.num_channels[-1] * self.window_size, self.output_size)
        
        # Initialize the final linear layer with small weights for stability
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.linear.bias, 0.0)
        
        print(f"\nTCN parameter #: {sum(p.numel() for p in self.tcn.parameters())}")
        print(f"FCNN parameter #: {sum(p.numel() for p in self.linear.parameters())}")
        print(f"Normalization: {self.norm}, Activation: {self.activation}")

    def forward(self, x):
        # x shape: (batch_size, input_size, time window size = sequence length)
        y = self.tcn(x)
        # Flatten the output from the TCN layer
        y = y.flatten(start_dim=1)  # Shape: (batch_size, num_channels[-1] * sequence_length)
        y = self.linear(y)
        return y