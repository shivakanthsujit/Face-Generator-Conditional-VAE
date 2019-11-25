import torch
from torch import nn

image_size = 64
hidden_size = 1024
latent_size = 128
columns = "Black_Hair	Blond_Hair	Brown_Hair	Male	No_Beard	Smiling	Straight_Hair	Wavy_Hair	Young"
columns = columns.split("\t")
num_columns = len(columns)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 1024, 1, 1)


class VAE(nn.Module):
    def __init__(
        self,
        image_channels=3,
        image_dim=image_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_classes=num_columns,
    ):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2),
            nn.LeakyReLU(0.2),
            Flatten(),
        )
        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, latent_size - num_classes)
        self.fc4 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size - num_classes)
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(hidden_size, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 6, 2),
            nn.Sigmoid(),
        )

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x, a):
        x = self.encoder(x)
        x = self.fc1(x)
        x = torch.cat((x, a), 1)
        x = self.fc2(x)

        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        z = self.sample(log_var, mean)

        z = self.fc3(z)
        z = torch.cat((z, a), 1)
        x = self.fc4(z)
        x = self.decoder(x)

        return x, mean, log_var
