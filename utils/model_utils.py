import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet3D(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_convs_per_block:int=2,
                 block_config:tuple=(24, 48, 96, 192), #switch this back to my format later
                 activation_type:str='ReLU',
                 pooling_type:str='MaxPool3d',
                 skip:bool=True,
                 **kwargs
    ):
        super(UNet3D, self).__init__()

        self.block_config = list(block_config)
        self.n_blocks = len(block_config)
        self.skip = skip

        self.activation_type = activation_type if callable(getattr(nn, activation_type)) \
            else Exception("Invalid activation_type (not an attribute of torch.nn")
        self.pooling_type = pooling_type if callable(getattr(nn, pooling_type)) \
            else Exception("Invalid pooling_type (not an attribute of torch.nn")

        
        # Encoding blocks:
        self.encoding = nn.Sequential()
        encoding_config = [in_channels] + self.block_config
        
        for b in range(0, len(encoding_config)-1):
            block = _UNet3D_Block(n_input_features=encoding_config[b],
                                  n_output_features=encoding_config[b+1],
                                  n_layers=n_convs_per_block,
                                  activation_type=activation_type,
                                  level=b,
                                  norm=True,
                                  drop=0,
            )
            self.encoding.add_module('ConvBlock%d' % (b+1), block)
            pool = getattr(nn, pooling_type)(kernel_size=2,
                                             stride=2,
                                             return_indices=True
            )
            self.encoding.add_module('Pool%d' % (b+1), pool)

            
        #Decoding blocks:
        self.decoding = nn.Sequential()
        decoding_config = [out_channels] + self.block_config
        
        for b in reversed(range(0, len(decoding_config) - 1)):
            upsample = nn.MaxUnpool3d(kernel_size=2, stride=2)
            self.decoding.add_module('Upsample%d' % (b+1), upsample)
            block = _UNet3D_BlockTranspose(n_input_features=decoding_config[b+1],
                                           n_output_features=decoding_config[b],
                                           n_layers=n_convs_per_block,
                                           activation_type=activation_type,
                                           level=b,
                                           norm=True,
                                           drop=0,
            )
            self.decoding.add_module('ConvBlock%d' % (b+1), block)
            
        
        last = nn.Sequential(nn.ConvTranspose3d(decoding_config[1], 
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                stride=1),
                             nn.Softmax(dim=1)
        )
        #self.decoding.add_module('LastStep', nn.Softmax(dim=1))
        self.decoding.add_module('LastConv', last)
        


    def forward(self, x):
        # Initialize intermediate steps
        enc = [None] * (self.n_blocks) # encoding
        dec = [None] * (self.n_blocks) # decoding
        idx = [None] * (self.n_blocks) # maxpool indices
        siz = [None] * (self.n_blocks) # maxunpool output size

        printout = False
        
        #breakpoint()
        # Encoding
        for b in range(0, self.n_blocks):
            if printout:  print('---------')
            if printout:  print('Encoding block %d' % (b+1))
            if printout:  print('---------')
            x = enc[b] = self.encoding.__getattr__('ConvBlock%d' % (b+1))(x)
            if printout:  print(self.encoding.__getattr__('ConvBlock%d' % (b+1)))
            siz[b] = x.shape
            if b != self.n_blocks - 1:
                x, idx[b] =  self.encoding.__getattr__('Pool%d' % (b+1))(x)
                if printout:  print(self.encoding.__getattr__('Pool%d' % (b+1)))
                if printout:  print(' ')
            
        # Decoding
        for b in reversed(range(0, self.n_blocks)):
            if printout:  print('---------')
            if printout:  print('Decoding block %d' % (b+1))
            if printout:  print('---------')
            if b != self.n_blocks - 1:
                x = self.decoding.__getattr__('Upsample%d' % (b+1))(x, idx[b], output_size=siz[b])
                if printout:  print(self.decoding.__getattr__('Upsample%d' % (b+1)))
            x = dec[b] = self.decoding.__getattr__('ConvBlock%d' % (b+1))(torch.cat([x, enc[b]], 1))
            if printout:  print('w/ skip', self.decoding.__getattr__('ConvBlock%d' % (b+1)))
            if printout:  print(' ')
            
        #x = self.decoding.LastConv(x)
        #if printout:  print(self.decoding.LastConv)
        if printout:  breakpoint()
        return x

            
        
class _UNet3D_Layer(nn.Module):
    def __init__(self,
                 n_input_features:int,
                 n_output_features:int,
                 activation_type:str,
                 norm=True,
                 drop=0,
                 activ=True,
                 **kwargs
    ):
        super(_UNet3D_Layer, self).__init__()

        self.add_module('norm', nn.BatchNorm3d(n_input_features) if norm else nn.Identity())
        self.add_module('activ', nn.ELU(inplace=False) if activation_type is not None else nn.Identity())
        self.add_module('conv', nn.Conv3d(n_input_features, n_output_features, kernel_size=3, padding=1,
                                          bias=False if norm else True))
        self.add_module('drop', nn.Dropout3d(drop) if drop > 0 else nn.Identity())

    def forward(self, x):
        return self.drop(self.conv(self.activ(self.norm(x))))


    
class _UNet3D_LayerTranspose(nn.Module):
    def __init__(self,
                 n_input_features:int,
                 n_output_features:int,
                 activation_type:str,
                 norm=True,
                 drop=0,
                 **kwargs
    ):
        super(_UNet3D_LayerTranspose, self).__init__()
        
        self.add_module('drop', nn.Dropout3d(drop) if drop > 0 else nn.Identity())
        self.add_module('norm', nn.BatchNorm3d(n_input_features) if norm else nn.Identity())
        self.add_module('activ', nn.ELU(inplace=False) if activation_type is not None else nn.Identity())
        self.add_module('conv', nn.ConvTranspose3d(n_input_features, n_output_features, kernel_size=3, padding=1,
                                          bias=False if norm else True))
        
    def forward(self, x):
        return self.conv(self.activ(self.norm(self.drop(x))))




class _UNet3D_Block(nn.ModuleDict):
    def __init__(self,
                 n_input_features:int,
                 n_output_features:int,
                 n_layers:int,
                 activation_type:str,
                 level:int,
                 norm=True,
                 drop=0,
                 skip=False,
                 **kwargs
    ):
        super(_UNet3D_Block, self).__init__()

        layer = _UNet3D_Layer(n_input_features=n_input_features,
                              n_output_features=n_output_features,
                              activation_type=activation_type if level != 0 else None,
                              norm=norm,
                              drop=drop,
        )
        self.add_module('ConvLayer1', layer)

        for i in range(1, n_layers):
            growth = 1 + (skip and i == n_layers - 1)
            layer = _UNet3D_Layer(n_input_features=n_output_features,
                                  n_output_features=growth*n_output_features,
                                  activation_type=activation_type,
                                  norm=norm,
                                  drop=drop,
            )
            self.add_module('ConvLayer%d' % (i + 1), layer)

            
    def forward(self, x):
        for name, layer in self.items():
            x = layer(x)
        return x


    
class _UNet3D_BlockTranspose(nn.ModuleDict):
    def __init__(self,
                 n_input_features:int,
                 n_output_features:int,
                 activation_type:str,
                 n_layers:int,
                 level:int,
                 norm=True,
                 drop=0,
                 skip=True,
                 **kwargs
    ):
        super(_UNet3D_BlockTranspose, self).__init__()
        for i in reversed(range(1,n_layers)):
            growth = 1 + (skip and i == n_layers - 1)
            layer = _UNet3D_LayerTranspose(n_input_features=growth*n_input_features,
                                           n_output_features=n_input_features,
                                           activation_type=activation_type,
                                           norm=norm,
                                           drop=drop,
            )
            self.add_module('ConvLayer%d' % (i + 1), layer)

        layer = _UNet3D_LayerTranspose(n_input_features=n_input_features,
                                       n_output_features=n_output_features, # if level > 0 else n_input_features,
                                       activation_type=activation_type,
                                       norm=norm,
                                       drop=drop,
        )
        self.add_module('ConvLayer1', layer)


    def forward(self, x):
        for name, layer in self.items():
            x = layer(x)
        return x
        
