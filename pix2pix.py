import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os

# 1. Define the Generator (U-Net architecture)
class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGenerator, self).__init__()
        def down_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up_block(in_feat, out_feat, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_feat),
                nn.ReLU()
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.encoder = nn.ModuleList([
            down_block(in_channels, 64, normalize=False),
            down_block(64, 128),
            down_block(128, 256),
            down_block(256, 512),
            down_block(512, 512),
            down_block(512, 512),
            down_block(512, 512)
        ])

        self.decoder = nn.ModuleList([
            up_block(512, 512, dropout=True),
            up_block(1024, 512, dropout=True),
            up_block(1024, 512, dropout=True),
            up_block(1024, 256),
            up_block(512, 128),
            up_block(256, 64)
        ])

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)

        skips = skips[:-1][::-1]  # Reverse skip connections (exclude bottleneck)
        for idx, up in enumerate(self.decoder):
            x = up(x)
            x = torch.cat([x, skips[idx]], dim=1)  # Concatenate skip connections

        return self.final(x)


# 2. Define the Discriminator (PatchGAN)
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(PatchGANDiscriminator, self).__init__()
        def block(in_feat, out_feat, stride=2, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=stride, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, 64, normalize=False),
            block(64, 128),
            block(128, 256),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        combined = torch.cat([img_A, img_B], dim=1)  # Concatenate input and target images
        return self.model(combined)

# 3. Define Dataset and DataLoader
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(os.listdir(root))
        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img[:,:256], img[:,256:]

    def __len__(self):
        return len(self.files)

# 4. Training
def train():
    # Hyperparameters
    epochs = 200
    lr = 2e-4
    batch_size = 16
    img_size = 256
    lambda_recon = 100

    # Model
    generator = UNetGenerator(3, 3).to(device)
    discriminator = PatchGANDiscriminator(6).to(device)

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # DataLoader
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size * 2)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = ImageDataset(root="data/folder", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for i, (img_A, img_B) in enumerate(dataloader):
            img_A, img_B = img_A.to(device), img_B.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            fake_B = generator(img_A)
            pred_fake = discriminator(img_A, fake_B)
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_pixel = criterion_pixelwise(fake_B, img_B)
            loss_G = loss_GAN + lambda_recon * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(img_A, img_B)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            pred_fake = discriminator(img_A, fake_B.detach())
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

            if i % 100 == 0:
                save_image(fake_B, f"output/fake_{epoch}_{i}.png")

# Start training
device = "cuda" if torch.cuda.is_available() else "cpu"
train()
