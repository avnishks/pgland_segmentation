import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the downsampling (encoding) block
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(inplace=True)
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv(x)
        return self.pool(conv), conv

# Define the upsampling (decoding) block
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.up(x)
        # Concatenate on the channels axis
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)

# Define the U-Net model
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.down1 = Down(in_channels, 24)
        self.down2 = Down(24, 24 * 2)
        self.down3 = Down(24 * 2, 24 * 4)
        self.down4 = Down(24 * 4, 24 * 8)
        self.up1 = Up(24 * 8, 24 * 4)
        self.up2 = Up(24 * 4, 24 * 2)
        self.up3 = Up(24 * 2, 24)
        self.conv = nn.Conv3d(24, out_channels, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, _ = self.down4(x)
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        return self.conv(x)
