import torch
import torch.nn as nn
import torch.nn.functional as F



# Define the U-Net model
class UNet3D(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_features_start:int=24, # this could just be hard coded? depends if n=24 is valid
                 n_blocks:int=4,
                 n_convs_per_block:int=2,
                 activation_type:str='ReLU',
                 pooling_type:str='MaxPool3d',
                 **kwargs):
        super(UNet3D, self).__init__()
        
        # Validate arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features_start = n_features_start
        self.n_blocks = n_blocks
        self.n_convs_per_block = n_convs_per_block

        self.activation_type = activation_type if callable(getattr(nn, activation_type)) \
            else Exception("Invalid activation_type (not an attribute of torch.nn")
        self.pooling_type = pooling_type if callable(getattr(nn, pooling_type)) \
            else Exception("Invalid pooling_type (not an attribute of torch.nn")
        
        self.blocks = nn.Sequential()

        
        # Encoding blocks (down = conv + activation --> pooling)
        n_input_features = in_channels
        n_output_features = n_features_start
        
        for b in range(0, n_blocks):
            if b == n_blocks - 1:
                block = _UNet3D_DownBlock(in_channels=n_input_features,
                                          out_channels=n_output_features,
                                          n_convs_per_block=n_convs_per_block,
                                          activation_type=activation_type,
                                          pooling_type=pooling_type,
                                          pool=False,
                )
            else:
                block = _UNet3D_DownBlock(in_channels=n_input_features,
                                          out_channels=n_output_features,
                                          n_convs_per_block=n_convs_per_block,
                                          activation_type=activation_type,
                                          pooling_type=pooling_type,
                                          pool=True,
                )
            self.blocks.add_module('down%d' % (b+1), block)
            n_input_features = n_output_features
            n_output_features = n_output_features * 2
        
            
        # Decoding blocks (up = up_conv --> concatenate w/ skip --> conv + activation)
        n_output_features = n_input_features // 2

        for b in range(0, n_blocks):
            if b == n_blocks - 1:
                block = nn.Sequential(nn.Conv3d(n_input_features,
                                                out_channels,
                                                kernel_size=1),
                                      nn.Softmax(dim=1))

            else:
                block = _UNet3D_UpBlock(in_channels=n_input_features,
                                        out_channels=n_output_features,
                                        n_convs_per_block=n_convs_per_block,
                                        activation_type=activation_type
                )
            self.blocks.add_module('up%d' % (b+1), block)
            n_input_features = n_output_features
            n_output_features = n_output_features // 2

            

    def forward(self, x):
        # Initialize intermediate steps
        encoding = [None] * (self.n_blocks + 1)
        decoding = [None] * (self.n_blocks + 1)
        skips = [None] * (self.n_blocks - 1)
        
        # Down
        encoding[0] = x
        for b in range(0, self.n_blocks):
            #print("Down", b+1)
            #print("input:", encoding[b].shape)
            #print(self.blocks.__getattr__('down%d' % (b+1)))
            if b != self.n_blocks - 1:
                encoding[b+1], skips[b] = self.blocks.__getattr__('down%d' % (b+1))(encoding[b])
            else:
                encoding[b+1], _ = self.blocks.__getattr__('down%d' % (b+1))(encoding[b])
            #print("output:", encoding[b+1].shape)
            #print(" ")
                       
        # Up
        decoding[0] = encoding[-1]
        for b in range(0, self.n_blocks):
            #print("Up", b+1)
            #print(self.blocks.__getattr__('up%d' % (b+1)))
            #print("input:", decoding[b].shape)
            if b != self.n_blocks - 1:
                skip_ind = self.n_blocks - b - 2
                #print("skip:", skips[skip_ind].shape)
                decoding[b+1] = self.blocks.__getattr__('up%d' % (b+1))(decoding[b], skips[skip_ind])
            else:
                decoding[b+1] = self.blocks.__getattr__('up%d' % (b+1))(decoding[b])
            #print("output:", decoding[b+1].shape)
            #print(" ")

        #breakpoint()
        return decoding[-1]
                


# Downsampling (encoding) block
class _UNet3D_DownBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_convs_per_block:int,
                 activation_type:str,
                 pooling_type:str,
                 pool=True,
    ):
        super(_UNet3D_DownBlock, self).__init__()
        self.n_convs = n_convs_per_block
        self.pool = pool
        
        # Define components
        self.pooling = getattr(nn, pooling_type)(kernel_size=2, stride=2)
        self.conv_block = nn.Sequential()

        for n in range(0, n_convs_per_block):
            if n == 0:
                layer = nn.Sequential(nn.Conv3d(in_channels,
                                                out_channels,
                                                kernel_size=3,
                                                padding=1),
                                      getattr(nn, activation_type)(inplace=True)
                )
            else:
                layer = nn.Sequential(nn.Conv3d(out_channels,
                                                out_channels,
                                                kernel_size=3,
                                                padding=1),\
                                      getattr(nn, activation_type)(inplace=True)
                )
            self.conv_block.add_module('convlayer%d' % (n+1), layer)


    def forward(self, x):
        for n in range(0, self.n_convs):
            x = self.conv_block.__getattr__('convlayer%d' % (n+1))(x)

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
                 activation_type:str
    ):
        super(_UNet3D_UpBlock, self).__init__()
        self.n_convs = n_convs_per_block
        
        # Define components
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential()
        
        for n in range(0,n_convs_per_block):    
            if n == 0:
                layer = nn.Sequential(nn.ConvTranspose3d(in_channels=in_channels,
                                                         out_channels=out_channels,
                                                         kernel_size=3,
                                                         padding=1),
                                      getattr(nn, activation_type)(inplace=True)
                )
            else:
                layer = nn.Sequential(nn.ConvTranspose3d(in_channels=out_channels,
                                                         out_channels=out_channels,
                                                         kernel_size=3,
                                                         padding=1),
                                      getattr(nn, activation_type)(inplace=True)
                )
            self.conv_block.add_module('convlayer%d' % (n+1), layer)
        

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        
        for n in range(0, self.n_convs):
            x = self.conv_block.__getattr__('convlayer%d' % (n+1))(x)
            
        return x
