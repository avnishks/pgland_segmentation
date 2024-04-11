import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding='same', activation='relu', batch_norm=None, conv_kwargs=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.ELU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, input_shape, nb_features, nb_levels, nb_labels, feat_mult=1, batch_norm=None, activation='relu'):
        super().__init__()
        self.input_shape = input_shape
        self.nb_features = nb_features
        self.nb_levels = nb_levels
        self.feat_mult = feat_mult
        self.batch_norm = batch_norm
        self.activation = activation

        # Encoding path
        self.encoder = nn.ModuleList()
        in_channels = self.input_shape[0]
        for level in range(self.nb_levels):
            out_channels = int(self.nb_features * (self.feat_mult ** level))
            self.encoder.append(ConvBlock(in_channels, out_channels, padding='same', activation=self.activation, batch_norm=self.batch_norm))
            in_channels = out_channels

        # Decoding path
        self.decoder = nn.ModuleList()
        for level in reversed(range(self.nb_levels-1)):
            out_channels = int(self.nb_features * (self.feat_mult ** level))
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
                ConvBlock(out_channels * 2, out_channels, padding='same', activation=self.activation, batch_norm=self.batch_norm)
            ))
            in_channels = out_channels

        # Classification layer
        self.classifier = nn.Conv3d(self.nb_features, nb_labels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoding path
        for level in range(self.nb_levels):
            x = self.encoder[level](x)
            if level < self.nb_levels - 1:
                skip_connections.append(x)
                x = nn.MaxPool3d(2)(x)

        # Decoding path
        for level in range(self.nb_levels-1):
            x = self.decoder[level][0](x)  # ConvTranspose3d
            skip_connection = skip_connections.pop()
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder[level][1](x)  # ConvBlock

        # Classification layer
        x = self.classifier(x)

        # Ensure output has the same spatial dimensions as the input
        # if x.shape[2:] != self.input_shape[1:]:
        #     x = nn.functional.interpolate(x, size=self.input_shape[1:], mode='trilinear', align_corners=True)

        return x