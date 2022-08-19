from torch import nn


class GAEExtractor(nn.Module):

    def __init__(self, encoder, decoder):
        super(GAEExtractor, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, opt, adj):
        z = self.encoder(opt=opt, adj=adj)
        opt_recon, adj_recon = self.decoder(z)
        return opt_recon, adj_recon, z
