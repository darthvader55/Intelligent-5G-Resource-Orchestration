
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
# %pip install dtaidistance
from dtaidistance import dtw

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""#1.Temporal Convolutional Network

##1(a). TCN Class Definition
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        # first dilated causal conv
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) * dilation, dilation=dilation)
        )
        # second dilated causal conv
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) * dilation, dilation=dilation)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # optional 1x1 conv to match channels
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        # 1x1 conv for channel mismatch
        if self.downsample is not None:
            residual = self.downsample(residual)
        # cropping output to match residual length (sequence dimension)
        if out.size(2) != residual.size(2):
            out = out[:, :, :residual.size(2)]
        return self.relu(out + residual)

class DilatedTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3, num_blocks=3, dropout=0.2):
        super(DilatedTCN, self).__init__()
        layers = []
        for i in range(num_blocks):
            dilation = 2 ** i  # exponential dilation
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(ResidualBlock(in_channels, hidden_dim, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, features) --> conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # taking last time step of output layer
        out = out[:, :, -1]
        return self.fc(out)

"""##1(b).Training & Evaluation Functions"""

def train_evaluate_seqmodel(model_architecture, X_train, y_train, X_val, y_val, X_test, y_test,
                       num_epochs=30, lr=1e-3, val_interval=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move data to device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    input_dim=X_train.shape[-1]

    # Instantiating model --- with best config after hyperparameter & model architecture tuning
    if model_architecture == DilatedTCN:
        model = DilatedTCN(
            input_dim,
            hidden_dim=16,
            output_dim=1,
            kernel_size=3,
            num_blocks=3,
            dropout=0.2
        ).to(device)

    elif model_architecture == BiLSTMRegressor:
        model = BiLSTMRegressor(
            input_dim,
            hidden_dim=16,
            num_layers=1,
            output_dim=1
        ).to(device)

    elif model_architecture == RNNRegressor:
        model = RNNRegressor(
            input_dim,
            hidden_dim=16,
            num_layers=1,
            output_dim=1
        ).to(device)

    elif model_architecture == GRURegressor:
        model = GRURegressor(
            input_dim,
            hidden_dim=16,
            num_layers=1,
            output_dim=1
        ).to(device)

    else: raise ValueError("Invalid model architecture")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    val_epochs = []


    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train).squeeze()
        train_loss = criterion(pred, y_train)
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        # Validation loss every val_interval epochs
        val_interval=3
        if epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).squeeze()
                val_loss = criterion(val_pred, y_val)
                val_losses.append(val_loss.item())
                val_epochs.append(epoch)

            print(f'Epoch {epoch}/{num_epochs} - Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    # Final evaluation on test set
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).squeeze()

        mse = mse_criterion(preds, y_test.squeeze()).item()
        mae = mae_criterion(preds, y_test.squeeze()).item()

    # Compute DTW distance
    dtw_dist = dtw.distance(y_test.cpu().numpy().flatten(), preds.cpu().numpy().flatten())

    print(f'TCN MSE: {mse:.4f}')
    print(f'TCN MAE: {mae:.4f}')
    print(f'TCN DTW Distance: {dtw_dist:.4f}')

    # Plot Loss Curves
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.rcParamsDefault['axes.prop_cycle'].by_key()['color'])
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(val_epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TCN Training & Validation Loss')
    plt.legend()
    plt.savefig('TCN_Train_Val_Loss.png', dpi=300)
    plt.show()

    # Plot Actual vs Predicted
    plt.figure(figsize=(7, 5))
    plt.plot(y_test.cpu(), label='Actual')
    plt.plot(preds.cpu(), label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Resource Allocation')
    plt.title(f'Actual vs Predicted Resource Allocation')
    plt.legend()
    plt.savefig(f'TCN_Actual_vs_Predicted.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model

# function for creating labels for synthetic data using the trained TCN model

def label_generator(model, X_synth):
    # Convert to torch tensor and move to device
    X_synth_tensor = torch.from_numpy(X_synth).float().to(device)

    # Predict synthetic labels
    model.eval()
    with torch.no_grad():
        y_synth = model(X_synth_tensor).squeeze()

    print(f"y_synth shape: {y_synth.shape}")
    return y_synth

# function for preparing synthetic & real data for different training configurations, namely: T(S+R)TR, TSTR, and TRTS.

def split_synth_data(X_train, X_synth, y_train, y_synth):
    # Train-val-test split
    train_size = int(0.7 * len(X_synth))
    val_size = int(0.1 * len(X_synth))

    # for use in T(S+R)TR & TSTR
    X_train_synth = X_synth[:train_size]
    y_train_synth = y_synth[:train_size]

    X_val_synth = X_synth[train_size:train_size + val_size]
    y_val_synth = y_synth[train_size:train_size + val_size]

    # for use in TRTS only
    X_test_synth = X_synth[train_size + val_size:]
    y_test_synth = y_synth[train_size + val_size:]

    ##---combining synthetic and real data---##
    ##############
    X_train_synth = torch.from_numpy(X_train_synth).float().to(device)
    y_train_synth = y_train_synth.float().to(device)

    X_val_synth = torch.from_numpy(X_val_synth).float().to(device)
    y_val_synth = y_val_synth.float().to(device)

    X_test_synth = torch.from_numpy(X_test_synth).float().to(device)
    y_test_synth = y_test_synth.float().to(device)
    ############

    X_combined = torch.cat((X_train, X_train_synth), dim=0)
    y_combined = torch.cat((y_train.squeeze(), y_train_synth), dim=0)

    # Shuffle window order
    perm = torch.randperm(X_combined.size(0))
    X_combined = X_combined[perm]
    y_combined = y_combined[perm]

    # Train-val-test split
    train_size = int(0.8 * len(X_combined))
    val_size = int(0.2 * len(X_combined))

    X_train_combined = X_combined[:train_size]
    y_train_combined = y_combined[:train_size]

    X_val_combined = X_combined[train_size:train_size + val_size]
    y_val_combined = y_combined[train_size:train_size + val_size]

    return (X_train_synth, y_train_synth,
            X_val_synth, y_val_synth,
            X_test_synth, y_test_synth,

            X_train_combined, y_train_combined,
            X_val_combined, y_val_combined)

"""#2.Recurrent Neural Networks

##2(a).BiLSTM
"""

class BiLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # Last layer forward and backward hidden states
        forward_hidden = h_n[-2]   # last layer forward
        backward_hidden = h_n[-1]  # last layer backward
        h = torch.cat((forward_hidden, backward_hidden), dim=1)
        return self.fc(h)

"""##2(b).Standard RNN"""

class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h_n = self.rnn(x)
        out = h_n[-1]
        return self.fc(out)

"""##2(c).GRU"""

class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        out = h_n[-1]
        return self.fc(out)
