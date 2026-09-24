from cbam.attentions import *

class CBAM(nn.Module):

    def __init__(self, channels : int, reduction : int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x
    
