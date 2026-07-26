"""
Generative Adversarial Network (GAN) — PyTorch Implementation
================================================================
Trains a simple GAN on the MNIST handwritten digit dataset.

Architecture:
    Generator     : noise vector (latent_dim) -> 28x28 fake image
    Discriminator : 28x28 image -> real/fake probability

Run:
    python gan.py

Outputs:
    - ./gan_output/generated_epoch_XX.png  (sample grids saved periodically)
    - ./gan_output/generator.pth           (final trained generator weights)
    - ./gan_output/discriminator.pth       (final trained discriminator weights)
    - ./gan_output/loss_curve.png          (generator/discriminator loss over training)
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)


# ----------------------------------------------------------------------------
# Generator Network
# ----------------------------------------------------------------------------
class Generator(nn.Module):
    """
    Maps a latent noise vector z ~ N(0, 1) of size `latent_dim`
    to a synthetic image of shape (1, 28, 28).
    """

    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        img_size = int(torch.prod(torch.tensor(img_shape)))

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, img_size),
            nn.Tanh(),  # output pixels scaled to [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


# ----------------------------------------------------------------------------
# Discriminator Network
# ----------------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    Maps an image of shape (1, 28, 28) to a single scalar in [0, 1]
    representing the estimated probability that the image is real.
    """

    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        img_size = int(torch.prod(torch.tensor(img_shape)))

        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# ----------------------------------------------------------------------------
# Training routine
# ----------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data -----------------------------------------------------------
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    dataset = datasets.MNIST(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    # ---- Models -----------------------------------------------------------
    img_shape = (1, 28, 28)
    generator = Generator(latent_dim=args.latent_dim, img_shape=img_shape).to(device)
    discriminator = Discriminator(img_shape=img_shape).to(device)

    # ---- Loss and optimizers ----------------------------------------------
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    g_losses, d_losses = [], []

    # ---- Training loop ------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)

            # Ground truth labels
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            real_imgs = imgs.to(device)

            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs = generator(z)

            # Generator wants the discriminator to classify its output as "valid"
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % args.log_interval == 0:
                print(
                    f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        # Save a sample grid of generated images each epoch
        with torch.no_grad():
            sample_z = torch.randn(25, args.latent_dim, device=device)
            sample_imgs = generator(sample_z)
            save_image(
                sample_imgs,
                os.path.join(args.output_dir, f"generated_epoch_{epoch:02d}.png"),
                nrow=5,
                normalize=True,
            )

    # ---- Save final models -------------------------------------------------
    torch.save(generator.state_dict(), os.path.join(args.output_dir, "generator.pth"))
    torch.save(
        discriminator.state_dict(), os.path.join(args.output_dir, "discriminator.pth")
    )

    # ---- Plot loss curves ---------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label="Generator loss")
    plt.plot(d_losses, label="Discriminator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    print(f"Training complete. Outputs saved to: {args.output_dir}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train a simple GAN on MNIST")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Adam learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of noise vector z")
    parser.add_argument("--log_interval", type=int, default=200, help="batches between log prints")
    parser.add_argument("--data_dir", type=str, default="./data", help="where to download/store MNIST")
    parser.add_argument("--output_dir", type=str, default="./gan_output", help="where to save results")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
