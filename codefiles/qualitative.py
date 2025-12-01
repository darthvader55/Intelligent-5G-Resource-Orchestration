
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""#1.tSNE Analysis"""

def tsne_function(X_train, X_synth):
    # data preparation for tsne analysis
    if isinstance(X_train, torch.Tensor):
            real_data = X_train.cpu().numpy()
    else:
        real_data = X_train

    synthetic_data = X_synth

    def reshape_for_analysis(data):
        """Reshape 3D sequence data to 2D for analysis"""
        return data.reshape(data.shape[0], -1)

    if isinstance(synthetic_data, torch.Tensor):
        synthetic_data = synthetic_data.cpu().numpy()

    real_flat = reshape_for_analysis(real_data)
    synth_flat = reshape_for_analysis(synthetic_data)

    # tsne analysis
    print("\n1. Performing t-SNE Analysis...")
    from sklearn.preprocessing import StandardScaler
    # Combine real and synthetic data for t-SNE
    combined_data = np.vstack([real_flat, synth_flat])
    labels = np.hstack([np.zeros(len(real_flat)), np.ones(len(synth_flat))])

    # === Scale the combined data ===
    scaler_tsne = StandardScaler()
    # scaler = MinMaxScaler()
    combined_data_scaled = scaler_tsne.fit_transform(combined_data)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=100, max_iter=3000) # 1000
    # tsne_results = tsne.fit_transform(combined_data)
    tsne_results = tsne.fit_transform(combined_data_scaled)


    # setting style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    # Separate plots for better visualization
    plt.figure(figsize=(7, 5))
    plt.scatter(tsne_results[labels==0, 0], tsne_results[labels==0, 1],
              alpha=0.6, label='Real Data', s=20, color='blue')
    plt.scatter(tsne_results[labels==1, 0], tsne_results[labels==1, 1],
              alpha=0.6, label='Synthetic Data', s=20, color='red')
    plt.title('t-SNE Projection of Real vs Synthetic Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.savefig('tSNE.png', dpi=300)
    plt.show()

"""#2.Window plots"""

feature_names = ['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']
feature_units = ['Value', 'dBm', 'ms', 'Mbps', 'Mbps']

def temporal_quality(X_train, X_synth, feature_names, feature_units, scaler, label_mapping):
    ##---data preparation for temporal quality analysis---##
    # Unscale X_train and X_synth back to original units
    X_train_2d = X_train.cpu().numpy().reshape(-1, len(feature_names)) # reshaping to 2D
    X_synth_2d = X_synth.reshape(-1, len(feature_names))

    # Apply inverse transform
    X_train_unscaled_2d = scaler.inverse_transform(X_train_2d)
    X_synth_unscaled_2d = scaler.inverse_transform(X_synth_2d)

    # Reshape back to (num_samples, seq_len, num_features)
    X_train_unscaled = X_train_unscaled_2d.reshape(X_train.shape[0], X_train.shape[1], len(feature_names))
    X_synth_unscaled = X_synth_unscaled_2d.reshape(X_synth.shape[0], X_synth.shape[1], len(feature_names))

    seq_len = X_train_unscaled.shape[1]
    ####

    # function for plotting temporal plots comparisons
    def plot_window(sample_idx, feature_idx, pic_num):
        real_data = X_train_unscaled[sample_idx, :, feature_idx]
        synthetic_data = X_synth_unscaled[sample_idx, :, feature_idx]
        unit = feature_units[feature_idx]

        plt.figure(figsize=(7, 5))
        plt.plot(range(seq_len), real_data, label='Real', color='blue', marker='o', markersize=4)
        plt.plot(range(seq_len), synthetic_data, label='Synthetic', color='red', marker='x', markersize=4)
        plt.title(f'Window {sample_idx} — {feature_names[feature_idx]} over time')
        plt.xlabel('Timestep')
        plt.xticks(np.arange(0, seq_len, 2))

        # Set ylabel
        if feature_idx == 0:
            plt.ylabel('Application_Type')
        else:
            plt.ylabel(f'{feature_names[feature_idx]} ({unit})')

        plt.legend()

        # If Application Type feature, replace y-tick labels with actual class names
        if feature_idx == 0:
            unique_labels = sorted(label_mapping.keys())
            plt.yticks(ticks=unique_labels, labels=[label_mapping[i] for i in unique_labels], fontsize=8)

        plt.savefig(f'Temporal_Quality_{feature_names[feature_idx]}_pic{pic_num}_unscaled.png', dpi=300, bbox_inches='tight')
        plt.show()

    # plotting
    # Application Type
    print('# ===== Application Type ===== #')
    plot_window(sample_idx=66, feature_idx=0, pic_num=1)
    # plot_window(sample_idx=149, feature_idx=0, pic_num=2)
    # plot_window(sample_idx=100, feature_idx=0, pic_num=3)

    # Signal Strength
    print('# ===== Signal Strength ===== #')
    plot_window(sample_idx=50, feature_idx=1, pic_num=1)
    # plot_window(sample_idx=202, feature_idx=1, pic_num=2)
    # plot_window(sample_idx=100, feature_idx=1, pic_num=3)


    # Latency
    print('# ===== Latency ===== #')
    plot_window(sample_idx=29, feature_idx=2, pic_num=1)
    # plot_window(sample_idx=260, feature_idx=2, pic_num=2)
    # plot_window(sample_idx=100, feature_idx=2, pic_num=3)

    # Required Bandwidth
    print('# ===== Required Bandwidth ===== #')
    plot_window(sample_idx=12, feature_idx=3, pic_num=1)
    # plot_window(sample_idx=250, feature_idx=3, pic_num=2)
    # plot_window(sample_idx=100, feature_idx=3, pic_num=3)

    # Allocated Bandwidth
    print('# ===== Allocated Bandwidth ===== #')
    plot_window(sample_idx=12, feature_idx=4, pic_num=1)
    # plot_window(sample_idx=76, feature_idx=4, pic_num=2)
    # plot_window(sample_idx=100, feature_idx=4, pic_num=3)



