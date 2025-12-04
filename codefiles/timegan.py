"""Time-Aware Generative Adversarial and Temporal Convolutional Learning for Intelligent 5G Resource Orchestration

Code author: Linus Ngatia (linusngatia434@gmail.com)"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""#1.TimeGAN Components

###Embedder, Recovery, Generator, Discriminator

*Using sigmoid activation function for all components - This led to signficant mode collapse*
"""

# # Components - all sigmoid

# class Embedder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
#         super(Embedder, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
#         self.linear = nn.Linear(hidden_dim, latent_dim)
#         self.activation = nn.Sigmoid() #########

#     def forward(self, x):
#         # x shape: (batch, seq_len, input_dim)
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
#         c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
#         out, _ = self.lstm(x, (h_0, c_0))

#         out = self.linear(out)
#         out = self.activation(out)
#         return out

# class Recovery(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, input_dim, num_layers=2):
#         super(Recovery, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
#         self.linear = nn.Linear(hidden_dim, input_dim)
#         self.activation = nn.Sigmoid()  #########

#     def forward(self, h):
#         # h shape: (batch, seq_len, latent_dim)
#         h_0 = torch.zeros(self.num_layers, h.size(0), self.hidden_dim).to(h.device)
#         c_0 = torch.zeros(self.num_layers, h.size(0), self.hidden_dim).to(h.device)
#         out, _ = self.lstm(h, (h_0, c_0))
#         out = self.linear(out)
#         out = self.activation(out)
#         return out

# class Generator(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, num_layers=2):
#         super(Generator, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
#         self.linear = nn.Linear(hidden_dim, latent_dim)
#         self.activation = nn.Sigmoid()  ###########

#     def forward(self, z):
#         # z shape: (batch, seq_len, latent_dim)
#         h_0 = torch.zeros(self.num_layers, z.size(0), self.hidden_dim).to(z.device)
#         c_0 = torch.zeros(self.num_layers, z.size(0), self.hidden_dim).to(z.device)
#         out, _ = self.lstm(z, (h_0, c_0))
#         out = self.linear(out)
#         out = self.activation(out)
#         return out

# class Supervisor(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, num_layers=2):
#         super(Supervisor, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
#         self.linear = nn.Linear(hidden_dim, latent_dim)
#         self.activation = nn.Sigmoid()   ###########

#     def forward(self, h):
#         # h shape: (batch, seq_len, latent_dim)
#         h_0 = torch.zeros(self.num_layers, h.size(0), self.hidden_dim).to(h.device)
#         c_0 = torch.zeros(self.num_layers, h.size(0), self.hidden_dim).to(h.device)
#         out, _ = self.lstm(h, (h_0, c_0))
#         out = self.linear(out)
#         out = self.activation(out)
#         return out

"""*Using tanh activation function for all components [except discriminator which should remain as sigmoid because its output is binary (real or fake)]*

Using tanh worked and was key in fixing mode collapse
"""

# tanh

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super(Embedder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.Tanh()  #######

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        out = self.activation(out)  #
        return out

class Recovery(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=2):
        super(Recovery, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()  ######

    def forward(self, h):
        out, _ = self.lstm(h)
        out = self.linear(out)
        out = self.activation(out) #
        return out

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers=2):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.Tanh()  ##########

    def forward(self, z):
        out, _ = self.lstm(z)
        out = self.linear(out)
        out = self.activation(out)  #
        return out

class Supervisor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers=2):
        super(Supervisor, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.Tanh()  #####

    def forward(self, h):
        out, _ = self.lstm(h)
        out = self.linear(out)
        out = self.activation(out)  #
        return out

"""###Discriminator

*Discriminator: Using last time-step for classification - led to mode collapse*
"""

# # Last time-step
# class Discriminator(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, num_layers=2):
#         super(Discriminator, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
#         self.linear = nn.Linear(hidden_dim, 1)
#         self.activation = nn.Sigmoid()

#     def forward(self, h):
#         # h shape: (batch, seq_len, latent_dim)
#         h_0 = torch.zeros(self.num_layers, h.size(0), self.hidden_dim).to(h.device)
#         c_0 = torch.zeros(self.num_layers, h.size(0), self.hidden_dim).to(h.device)
#         out, _ = self.lstm(h, (h_0, c_0))
#         # using only last time step
#         out = self.linear(out[:, -1, :])
#         out = self.activation(out)
#         return out

"""*Discriminator: Using mean pooling across timesteps in sequence - This helped in fixing mode collapse*"""

# Mean Pooling
class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers=2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, h):
        h_0 = torch.zeros(self.num_layers, h.size(0), self.hidden_dim).to(h.device)
        c_0 = torch.zeros(self.num_layers, h.size(0), self.hidden_dim).to(h.device)
        out, _ = self.lstm(h, (h_0, c_0))

        # pooling across timesteps (mean)
        out = torch.mean(out, dim=1)
        out = self.linear(out)
        out = self.activation(out)
        return out

"""#2.TimeGAN Training"""

# function for training TimeGAN model
def timegan_training (Embedder, Recovery, Generator, Supervisor, Discriminator, X_train,
                      input_dim, hidden_dim, latent_dim, num_layers,
                      lr_embedder, lr_recovery, lr_supervisor, lr_generator, lr_discriminator,
                      device):

    # Initialising networks
    embedder = Embedder(input_dim, hidden_dim, latent_dim, num_layers).to(device)
    recovery = Recovery(latent_dim, hidden_dim, input_dim, num_layers).to(device)
    generator = Generator(latent_dim, hidden_dim, num_layers).to(device)
    supervisor = Supervisor(latent_dim, hidden_dim, num_layers).to(device)
    discriminator = Discriminator(latent_dim, hidden_dim, num_layers).to(device)

    # Loss functions
    recon_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss()
    supervised_loss_fn = nn.MSELoss()

    # Optimizers
    # Phase 1 & 2
    optimizer_embedder = optim.Adam(embedder.parameters(), lr=lr_embedder, betas=(0.5, 0.999))
    optimizer_recovery = optim.Adam(recovery.parameters(), lr=lr_recovery, betas=(0.5, 0.999))
    optimizer_supervisor = optim.Adam(supervisor.parameters(), lr=lr_supervisor, betas=(0.5, 0.999))

    # Phase 3
    optimizer_generator = optim.Adam(generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999))

    ######################

    # Phase 1: Autoencoder pretraining
    print("Phase 1: Autoencoder Pretraining")
    pretrain_epochs = 200
    for epoch in range(pretrain_epochs):
        embedder.train()
        recovery.train()

        optimizer_embedder.zero_grad()
        optimizer_recovery.zero_grad()

        # Forward pass
        h = embedder(X_train)
        x_hat = recovery(h)

        # Compute reconstruction loss
        recon_loss = recon_loss_fn(x_hat, X_train)

        # Backward pass
        recon_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(recovery.parameters(), max_norm=1.0)

        optimizer_embedder.step()
        optimizer_recovery.step()

        if (epoch + 1) % 20 == 0:
            print(f"Pretrain Epoch [{epoch + 1}/{pretrain_epochs}], Recon Loss: {recon_loss.item():.6f}")

    # Phase 2: Supervisor pretraining
    print("\nPhase 2: Supervisor Pretraining")
    supervisor_epochs = 200
    for epoch in range(supervisor_epochs):
        embedder.train()
        supervisor.train()

        optimizer_embedder.zero_grad()
        optimizer_supervisor.zero_grad()

        # Forward pass
        h = embedder(X_train)

        # Supervisor predicts next timestep
        h_supervise_pred = supervisor(h[:, :-1, :])
        h_supervise_true = h[:, 1:, :]

        # Compute supervised loss
        supervised_loss = supervised_loss_fn(h_supervise_pred, h_supervise_true)

        # Backward pass
        supervised_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(supervisor.parameters(), max_norm=1.0)

        optimizer_embedder.step()
        optimizer_supervisor.step()

        if (epoch + 1) % 20 == 0:
            print(f"Supervisor Epoch [{epoch + 1}/{supervisor_epochs}], Supervised Loss: {supervised_loss.item():.6f}")

    # Phase 3: Joint adversarial training
    print("\nPhase 3: Joint Adversarial Training")
    gamma = 1.0
    eta = 10

    adv_epochs = 1000

    # storing losses for plotting
    gen_losses = []
    disc_real_losses = []
    disc_fake_losses = []
    recon_losses = []

    for epoch in range(adv_epochs):
        # Train Generator (G + S) multiple times (relative to discriminator)
        for _ in range(2):
            generator.train()
            supervisor.train()

            optimizer_generator.zero_grad()
            optimizer_supervisor.zero_grad()

            # Generate random noise
            batch_size = X_train.size(0)
            seq_len = X_train.size(1)
            z = torch.randn(batch_size, seq_len, latent_dim).to(device)

            # Generate fake sequences
            e_fake = generator(z)
            h_fake = supervisor(e_fake)

            # Discriminator predictions on fake data
            y_fake_h = discriminator(h_fake)
            y_fake_e = discriminator(e_fake)

            # Real labels for generator training
            valid_labels = torch.ones_like(y_fake_h)

            # Generator losses
            g_loss_u = bce_loss_fn(y_fake_h, valid_labels)
            g_loss_u_e = bce_loss_fn(y_fake_e, valid_labels)

            # Supervised loss (temporal consistency)
            g_loss_s = supervised_loss_fn(h_fake[:, :-1, :], e_fake[:, 1:, :])

            # Total generator loss
            gen_loss = g_loss_u + gamma * g_loss_u_e + eta * g_loss_s

            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(supervisor.parameters(), max_norm=1.0)

            optimizer_generator.step()
            optimizer_supervisor.step()

        # Train Embedder and Recovery
        embedder.train()
        recovery.train()

        optimizer_embedder.zero_grad()
        optimizer_recovery.zero_grad()

        # Real data reconstruction
        h_real = embedder(X_train)
        x_tilde = recovery(h_real)

        # Reconstruction loss
        recon_loss = recon_loss_fn(x_tilde, X_train)

        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(recovery.parameters(), max_norm=1.0)

        optimizer_embedder.step()
        optimizer_recovery.step()

        # Train Discriminator
        discriminator.train()
        optimizer_discriminator.zero_grad()

        with torch.no_grad():
            h_real = embedder(X_train)
            z = torch.randn(batch_size, seq_len, latent_dim).to(device)
            e_fake = generator(z)
            h_fake = supervisor(e_fake)

        # Discriminator predictions
        y_real = discriminator(h_real)
        y_fake_h = discriminator(h_fake)
        y_fake_e = discriminator(e_fake)

        # Labels
        real_labels = torch.ones_like(y_real)
        fake_labels = torch.zeros_like(y_fake_h)
        fake_labels_e = torch.zeros_like(y_fake_e)

        # Discriminator losses (separate real and fake)
        d_loss_real = bce_loss_fn(y_real, real_labels)
        d_loss_fake_h = bce_loss_fn(y_fake_h, fake_labels)
        d_loss_fake_e = bce_loss_fn(y_fake_e, fake_labels_e)

        # Combined fake loss for tracking
        d_loss_fake_combined = d_loss_fake_h + gamma * d_loss_fake_e

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake_combined

        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_discriminator.step()

        # Store losses for plotting
        gen_losses.append(gen_loss.item())
        disc_real_losses.append(d_loss_real.item())
        disc_fake_losses.append(d_loss_fake_combined.item())
        recon_losses.append(recon_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{adv_epochs}] "
                  f"Gen Loss: {gen_loss.item():.4f} "
                  f"Disc Real Loss: {d_loss_real.item():.4f} "
                  f"Disc Fake Loss: {d_loss_fake_combined.item():.4f} "
                  f"Recon Loss: {recon_loss.item():.6f}")

    # saving final checkpoint
    checkpoint_filename = 'checkpoint.pth'
    saved_checkpoint_path = save_checkpoint(
        adv_epochs, embedder, recovery, generator, discriminator, supervisor,
        optimizer_embedder, optimizer_recovery, optimizer_generator,
        optimizer_discriminator, optimizer_supervisor,
        filename=checkpoint_filename
    )
    # Save separate component files to root of session (./)
    save_path = "./"
    save_individual_components(embedder, recovery, generator, supervisor, discriminator, save_path=save_path)


    ####
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.rcParamsDefault['axes.prop_cycle'].by_key()['color'])
    plt.figure(figsize=(7, 5))
    plt.plot(gen_losses, label='Generator Loss')

    plt.plot(disc_fake_losses, label='Discriminator Fake Loss')
    plt.plot(disc_real_losses, label='Discriminator Real Loss')

    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("TimeGAN Training Losses")
    plt.legend()
    # plt.grid()
    plt.savefig(f'2b_TimeGAN_Training_Losses_lr_{lr_generator}.png', dpi=300)
    plt.show()


    print("\n\nTimeGAN training completed successfully.")
    return embedder, recovery, generator, discriminator, supervisor, saved_checkpoint_path

# function for saving trained components
def save_checkpoint(epoch, embedder, recovery, generator, discriminator, supervisor,
                    optimizer_embedder, optimizer_recovery, optimizer_generator,
                    optimizer_discriminator, optimizer_supervisor, filename):
    torch.save({
        'epoch': epoch,
        'embedder_state': embedder.state_dict(),
        'recovery_state': recovery.state_dict(),
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'supervisor_state': supervisor.state_dict(),
        'optimizer_embedder_state': optimizer_embedder.state_dict(),
        'optimizer_recovery_state': optimizer_recovery.state_dict(),
        'optimizer_generator_state': optimizer_generator.state_dict(),
        'optimizer_discriminator_state': optimizer_discriminator.state_dict(),
        'optimizer_supervisor_state': optimizer_supervisor.state_dict()
    }, filename)
    print(f"Checkpoint saved at epoch {epoch} to {filename}")
    return filename

# function to save individual component files
def save_individual_components(embedder, recovery, generator, supervisor, discriminator, save_path="./"):
    os.makedirs(save_path, exist_ok=True)
    torch.save(embedder.state_dict(), os.path.join(save_path, 'embedder.pth'))
    torch.save(recovery.state_dict(), os.path.join(save_path, 'recovery.pth'))
    torch.save(generator.state_dict(), os.path.join(save_path, 'generator.pth'))
    torch.save(supervisor.state_dict(), os.path.join(save_path, 'supervisor.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_path, 'discriminator.pth'))
    print(f"Individual model files saved to {os.path.abspath(save_path)}")
    return save_path

# function for reloading trained TimeGAN components
def reload_components(Embedder, Recovery, Generator, Supervisor, Discriminator,
                                   X_train, device, hidden_dim=16, latent_dim=16, num_layers=2,
                                   save_path="./"):
    input_dim = X_train.shape[2]

    # Initialize model instances
    embedder = Embedder(input_dim, hidden_dim, latent_dim, num_layers).to(device)
    recovery = Recovery(latent_dim, hidden_dim, input_dim, num_layers).to(device)
    generator = Generator(latent_dim, hidden_dim, num_layers).to(device)
    supervisor = Supervisor(latent_dim, hidden_dim, num_layers).to(device)
    discriminator = Discriminator(latent_dim, hidden_dim, num_layers).to(device)

    # Load saved weights
    embedder.load_state_dict(torch.load(os.path.join(save_path, 'embedder.pth'), map_location=device))
    recovery.load_state_dict(torch.load(os.path.join(save_path, 'recovery.pth'), map_location=device))
    generator.load_state_dict(torch.load(os.path.join(save_path, 'generator.pth'), map_location=device))
    supervisor.load_state_dict(torch.load(os.path.join(save_path, 'supervisor.pth'), map_location=device))
    discriminator.load_state_dict(torch.load(os.path.join(save_path, 'discriminator.pth'), map_location=device))

    # Set to device
    for model in [embedder, recovery, generator, supervisor, discriminator]:
        model.to(device)
        model.eval()  # set to eval mode (safe default)

    print("TimeGAN components loaded successfully from:", save_path)
    return embedder, recovery, generator, supervisor, discriminator

"""#3.Generating Synthetic Data"""

# function for generating synthetic data using the trained TimeGAN model
def generate_synthetic_data(generator, supervisor, recovery, num_samples, seq_len, latent_dim, device):
    generator.eval()
    supervisor.eval()
    recovery.eval()

    with torch.no_grad():
        # Generate random noise
        z = torch.randn(num_samples, seq_len, latent_dim).to(device)

        # Generate synthetic latent sequences
        e_fake = generator(z)
        h_fake = supervisor(e_fake)

        # Recover synthetic data
        synthetic_data = recovery(h_fake)

    return synthetic_data.cpu().numpy()
