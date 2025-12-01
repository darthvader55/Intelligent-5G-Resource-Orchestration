
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_loading(dataset):

    # loading in data
    df = pd.read_csv(f'{dataset}.csv')
    print("Data loaded successfully.")

    # removing units from values and converting to float
    df['Signal_Strength'] = df['Signal_Strength'].str.replace(' dBm', '').astype(float)
    df['Latency'] = df['Latency'].str.replace(' ms', '').astype(float)
    df['Resource_Allocation'] = df['Resource_Allocation'].str.replace('%', '').astype(float) / 100.0

    # converting all bandwidth values to Mbps
    def convert_bandwidth(value):
        if 'Kbps' in value:
            return float(value.replace(' Kbps', '')) / 1000   # changing to Mbps
        elif 'Mbps' in value:
            return float(value.replace(' Mbps', ''))
        else:
            raise ValueError(f"Unexpected unit in value: '{value}'")

    df['Required_Bandwidth'] = df['Required_Bandwidth'].apply(convert_bandwidth)
    df['Allocated_Bandwidth'] = df['Allocated_Bandwidth'].apply(convert_bandwidth)

    # box plots for diff QoS metrics
    units = {
        'Latency': 'ms',
        'Signal_Strength': 'dBm',
        'Required_Bandwidth': 'Mbps',
        'Allocated_Bandwidth': 'Mbps'
    }

    palette = sns.color_palette("Set2", n_colors=len(df['Application_Type'].unique()))

    print("\nBox plots for diffErent QoS metrics: ")
    for col in ['Latency', 'Signal_Strength', 'Required_Bandwidth', 'Allocated_Bandwidth']:
        plt.figure(figsize=(7,5))
        sns.boxplot(x='Application_Type', y=col, data=df, palette=palette, hue='Application_Type', legend=False)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.ylabel(f'{col} ({units[col]})')
        plt.title(f'{col}')
        plt.savefig(f'{col}_by_Application_Boxplot.png', dpi=300, bbox_inches='tight')
        plt.show()

    # encoding Application_Type
    le = LabelEncoder()
    df['Application_Type'] = le.fit_transform(df['Application_Type'])
    label_mapping = {index: label for index, label in enumerate(le.classes_)}

    # sorting by timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=False)
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    # features and labels
    features = df[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
    labels = df['Resource_Allocation']

    # feature scaling
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # creating window sequences
    sequence_length = 20
    X_seq = []
    y_seq = []
    for i in range(sequence_length, len(df)):
        X_seq.append(scaled_features[i-sequence_length:i])
        y_seq.append(labels.iloc[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # converting to tensors
    X = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1).to(device)

    # Train-val-test split
    train_size = int(0.7 * len(X))
    val_size = int(0.1 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Data shape: {X_train.shape}")

    print(".\n.\nDataset is ready")

    return df, label_mapping, scaler, X_train, y_train, X_val, y_val, X_test, y_test, sequence_length
