import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class ConvolutionalBlock(nn.Module):
  def __init__(self,
               in_channels, 
               out_channels,
               kernel_size
               ):
    super(ConvolutionalBlock, self).__init__()
    self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups=1, bias=False, padding_mode='zeros', 
                                        device=None, dtype=None),
                              nn.BatchNorm3d(out_channels),    # not used in original paper, but used in other papers and implementations, beneficial?
                              nn.ReLU(inplace = True),
                              nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups=1, bias=False, padding_mode='zeros', 
                                        device=None, dtype=None),
                              nn.BatchNorm3d(out_channels),    # not used in original paper, but used in other papers and implementations, beneficial?
                              nn.ReLU(inplace = True),
    )

  def forward(self, x):
    return self.conv(x)
  

class UNet(nn.Module):
  def __init__(self,
               in_channels = 1, out_channels = 1, #think about this?
               features = [32, 64, 128, 256]):
    super(UNet, self).__init__()
    self.ups = nn.ModuleList()    # store convolutional layers
    self.downs = nn.ModuleList()
    self.pool = nn.MaxPool3d(kernel_size = 2, stride = 2)     # use layer in between in forward method
    # 161 x 161, output: 160 x 160

    # Downpart of UNET
    # go through features
    for feature in features:
      self.downs.append(ConvolutionalBlock(in_channels, feature))     # add layer to module list, map input to feature
      in_channels = feature

    
    # Up part of UNET
    for feature in reversed(features):
      self.ups.append(nn.ConvTranspose3d(in_channels = feature*2, out_channels = feature, kernel_size = 2, stride = 2))
      self.ups.append(ConvolutionalBlock(feature*2, feature))

    self.bottleneck = ConvolutionalBlock(features[-1], features[-1]*2)
    self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size = 1)


  def forward(self, x):
    skip_connections = []

    for down in self.downs:
      x = down(x)
      skip_connections.append(x)    # first highest resolution
      x = self.pool(x)

    x = self.bottleneck(x)
    skip_connections = skip_connections[::-1]


  # upsampling and concatenating
    for index in range(0, len(self.ups), 2):
      x = self.ups[index](x)
      skip_connection = skip_connections[index//2]      # we're getting the skip connection and then we're concatenating

      if x.shape != skip_connection.shape:
        x = TF.resize(x, size = skip_connection.shape[2:])

      concat_skip = torch.cat((skip_connection, x), dim = 1)
      x = self.ups[index + 1](concat_skip)

    return self.final_conv(x)