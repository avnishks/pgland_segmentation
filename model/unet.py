import torch
import torch.nn as nn
import torch.nn.functional as F



class UNetXD(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_convs_per_block:int=2,
                 block_config:tuple=(24, 48, 96, 192),
                 normalization_type:str='Instance',
                 activation_type:str='ReLU',
                 skip:bool=True,
                 X:int=3,
                 **kwargs
    ):
        super(UNetXD, self).__init__()

        #self.n_dims = X
        self.block_config = list(block_config)
        self.n_blocks = len(block_config)
        self.skip = skip

        self.activation_type = activation_type if callable(getattr(nn, activation_type)) \
            else Exception("Invalid activation_type (not an attribute of torch.nn")

        
        # Encoding blocks:
        self.encoding = nn.Sequential()
        encoding_config = [in_channels] + self.block_config
        
        for b in range(0, len(encoding_config)-1):
            block = _UNetBlock(n_input_features=encoding_config[b],
                               n_output_features=encoding_config[b+1],
                               n_layers=n_convs_per_block,
                               norm_type=normalization_type,
                               activ_type=activation_type,
                               level=b,
                               drop=0,
                               X=X,
            )
            self.encoding.add_module('ConvBlock%d' % (b+1), block)
            pool = eval('nn.MaxPool%dd' % X)(kernel_size=2,
                                             stride=2,
                                             return_indices=True
            )
            self.encoding.add_module('Pool%d' % (b+1), pool)

            
        #Decoding blocks:
        self.decoding = nn.Sequential()
        decoding_config = [out_channels] + self.block_config
        
        for b in reversed(range(0, len(decoding_config) - 1)):
            upsample = eval('nn.MaxUnpool%dd' % X)(kernel_size=2, stride=2)
            self.decoding.add_module('Upsample%d' % (b+1), upsample)
            
            block = _UNetBlockTranspose(n_input_features=decoding_config[b+1],
                                        n_output_features=decoding_config[b],
                                        n_layers=n_convs_per_block,
                                        norm_type=normalization_type,
                                        activ_type=activation_type,
                                        level=b,
                                        drop=0,
                                        X=X,
            )
            self.decoding.add_module('ConvBlock%d' % (b+1), block)
                    

    def forward(self, x):
        enc = [None] * (self.n_blocks) # encoding
        dec = [None] * (self.n_blocks) # decoding
        idx = [None] * (self.n_blocks) # maxpool indices
        siz = [None] * (self.n_blocks) # maxunpool output size

        printout = False
        
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
            
        if printout:  breakpoint()
        return x

            
        
class _UNetLayer(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 norm_type:str=None,
                 activ_type:str=None,
                 drop=0,
                 **kwargs
    ):
        super(_UNetLayer, self).__init__()

        normXd = eval('nn.%sNorm%dd' % (norm_type, X))(n_input_features)
        activXd = eval('nn.%s' % activ_type)() if activ_type is not None else nn.Identity()
        convXd = eval('nn.Conv%dd' % X)(n_input_features,
                                        n_output_features,
                                        kernel_size=3,
                                        padding=1,
                                        bias=False if norm_type is not None else True
        )
        dropXd = eval('nn.Dropout%dd' % X)
        
        self.add_module('norm', normXd if norm_type is not None else nn.Identity())
        self.add_module('activ', activXd if activ_type is not None else nn.Identity())
        self.add_module('conv', convXd)
        self.add_module('drop', dropXd if drop > 0 else nn.Identity())

    def forward(self, x):
        return self.drop(self.conv(self.activ(self.norm(x))))


    
class _UNetLayerTranspose(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 norm_type:str=None,
                 activ_type:str=None,
                 drop=0,
                 **kwargs
    ):
        super(_UNetLayerTranspose, self).__init__()

        dropXd = eval('nn.Dropout%dd' % X)
        normXd = eval('nn.%sNorm%dd' % (norm_type, X))(n_input_features)
        activXd = eval('nn.%s' % activ_type)() if activ_type is not None else nn.Identity()
        convXd = eval('nn.ConvTranspose%dd' % X)(n_input_features,
                                                  n_output_features,
                                                  kernel_size=3,
                                                  padding=1,
                                                  bias=False if norm_type is not None else True
        )

        self.add_module('drop', dropXd if drop > 0 else nn.Identity())
        self.add_module('norm', normXd if norm_type else nn.Identity())
        self.add_module('activ', activXd) # if activation_type is not None else nn.Identity())
        self.add_module('conv', convXd)


    def forward(self, x):
        return self.conv(self.activ(self.norm(self.drop(x))))




class _UNetBlock(nn.ModuleDict):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 n_layers:int,
                 norm_type:str,
                 activ_type:str,
                 level:int,
                 drop=0,
                 skip=False,
                 **kwargs
    ):
        super(_UNetBlock, self).__init__()

        layer = _UNetLayer(n_input_features=n_input_features,
                           n_output_features=n_output_features,
                           norm_type=norm_type,
                           activ_type=activ_type if level != 0 else None,
                           drop=drop,
                           X=X,
        )
        self.add_module('ConvLayer1', layer)

        for i in range(1, n_layers):
            growth = 1 + (skip and i == n_layers - 1)
            layer = _UNetLayer(n_input_features=n_output_features,
                               n_output_features=growth*n_output_features,
                               norm_type=norm_type,
                               activ_type=activ_type,
                               drop=drop,
                               X=X,
            )
            self.add_module('ConvLayer%d' % (i + 1), layer)

            
    def forward(self, x):
        for name, layer in self.items():
            x = layer(x)
        return x


    
class _UNetBlockTranspose(nn.ModuleDict):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 n_layers:int,
                 norm_type:str,
                 activ_type:str,
                 level:int,
                 drop=0,
                 skip=True,
                 **kwargs
    ):
        super(_UNetBlockTranspose, self).__init__()
        for i in reversed(range(1,n_layers)):
            growth = 1 + (skip and i == n_layers - 1)
            layer = _UNetLayerTranspose(n_input_features=growth*n_input_features,
                                        n_output_features=n_input_features,
                                        norm_type=norm_type,
                                        activ_type=activ_type,
                                        drop=drop,
                                        X=X,
            )
            self.add_module('ConvLayer%d' % (i + 1), layer)

        layer = _UNetLayerTranspose(n_input_features=n_input_features,
                                    n_output_features=n_output_features, # if level > 0 else n_input_features,
                                    norm_type=norm_type,
                                    activ_type=activ_type,
                                    drop=drop,
                                    X=X,
        )
        self.add_module('ConvLayer1', layer)


    def forward(self, x):
        for name, layer in self.items():
            x = layer(x)
        return x
   


class UNet3D(UNetXD):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_convs_per_block=2,
                         block_config=(24, 48, 96),
                         normalization_type='Instance',
                         activation_type='ELU',
                         skip=True,
                         X=3,
                         **kwargs
        )

