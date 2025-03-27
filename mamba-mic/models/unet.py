from torch import nn


class UNet(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
