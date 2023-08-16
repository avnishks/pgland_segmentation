import torch
import torch.nn as nn
import torch.nn.functional as F



# Define the U-Net model
class UNet3D(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_features_start:int, # this could just be hard coded? depends if n=24 is valid
                 num_blocks:int,
                 num_convs_per_block:int, # can this be the same for each block? or should user be able to change it
                 activation_type:str,
                 pooling_type:str):
        super(UNet3D, self).__init__()
        
        # Validate arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features_start = n_features_start
        self.num_blocks = num_blocks
        self.num_convs_per_block = num_convs_per_block
        self.activation_type = activation_type
        self.pooling_type = pooling_type
        ### to do: check if nn.(activation_type) and nn.(pooling_type) exist, else throw error
        ############ (assert(callable(getattr(parent,'name'))))

        
        self.blocks = nn.Sequential()
                
        # Encoding blocks (down = conv + activation --> pooling)
        n_input_features = in_channels
        n_output_features = n_features_start

        for b in range(0, num_blocks):
            if b == num_blocks - 1:
                block = _UNet3D_DownBlock(n_input_features, n_output_features,
                                          num_convs_per_block, activation_type, pooling_type,
                                          pool=False)
            else:
                block = _UNet3D_DownBlock(n_input_features, n_output_features,
                                      num_convs_per_block, activation_type, pooling_type)
            self.blocks.add_module('down%d' % (b+1), block)
            n_input_features = n_output_features
            n_output_features = n_output_features * 2
        
            
        # Decoding blocks (up = up_conv --> concatenate w/ skip --> conv + activation)
        n_output_features = n_input_features // 2

        for b in range(0, num_blocks):
            if b == num_blocks - 1:
                block = nn.Conv3d(n_input_features, out_channels, kernel_size=1, padding=1)
            else:
                block = _UNet3D_UpBlock(n_input_features, n_output_features,
                                        num_convs_per_block, activation_type)
            self.blocks.add_module('up%d' % (b+1), block)
            n_input_features = n_output_features
            n_output_features = n_output_features // 2

            

    def forward(self, x):
        # Initialize intermediate steps
        encoding = [None] * self.num_blocks
        decoding = [None] * self.num_blocks
        skips = [None] * (self.num_blocks - 1)
        
        # Down
        for b in range(0, self.num_blocks):
            #print("Down", b+1)
            if b != self.num_blocks - 1:
                encoding[b], skips[b] = self.blocks.__getattr__('down%d' % (b+1))(x)
            else:
                encoding[b], _ = self.blocks.__getattr__('down%d' % (b+1))(x)

            x = encoding[b]
                       
        # Up
        for b in range(0, self.num_blocks):
            #print("Up", b+1)
            if b != self.num_blocks - 1:
                skip_ind = self.num_blocks - b - 2
                decoding[b] = self.blocks.__getattr__('up%d' % (b+1))(x, skips[skip_ind])
            else:
                decoding[b] = self.blocks.__getattr__('up%d' % (b+1))(x)

            x = decoding[b]

        return x
                


# Downsampling (encoding) block
class _UNet3D_DownBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_convs_per_block:int,
                 activation_type:str,
                 pooling_type:str,
                 pool=True):

        super(_UNet3D_DownBlock, self).__init__()
        self.n_convs = n_convs_per_block
        self.pool = pool
        
        # Define components
        self.pooling = getattr(nn, pooling_type)(kernel_size=2, stride=2)
        self.conv_block = nn.Sequential()

        for n in range(0, n_convs_per_block):
            if n == 0:
                layer = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),\
                                      getattr(nn, activation_type)(inplace=True))
            else:
                layer = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),\
                                      getattr(nn, activation_type)(inplace=True))
            self.conv_block.add_module('convlayer%d' % (n+1), layer)

            
    def forward(self, x):
        x = self.conv_block(x)
        if self.pool:
            return self.pooling(x), x
        else:
            return x, x


# Define the upsampling (decoding) block
class _UNet3D_UpBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_convs_per_block:int,
                 activation_type:str):
        super(_UNet3D_UpBlock, self).__init__()

        # Up-convolution
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        # Add user defined number of convolution layers
        self.conv_block = nn.Sequential()
        for n in range(0,n_convs_per_block):
            if n == 0:
                layer = nn.Sequential(
                    nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, padding=1),
                    getattr(nn, activation_type)(inplace=True))
            else:
                layer = nn.Sequential(
                    nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, padding=1),
                    getattr(nn, activation_type)(inplace=True))
            self.conv_block.add_module('convlayer%d' % (n+1), layer)
        


    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)

        return self.conv_block(x)

