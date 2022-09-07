from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
import torch


class Autoencoder(Module):
    def __init__(self, data_dim, latent):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential(
            Linear(data_dim, 128),
            ReLU(True),
            Linear(128, 64),
            ReLU(True), Linear(64, latent))
        self.decoder = Sequential(
            Linear(latent, 64),
            ReLU(True),
            Linear(64, 128),
            ReLU(True), Linear(128, data_dim))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


class LAutoencoder(Module):
    def __init__(self, data_dim, latent):
        super(LAutoencoder, self).__init__()
        self.encoder = Sequential(
            Linear(data_dim, 256),
            ReLU(True),
            Linear(256, 256),
            ReLU(True), Linear(256, latent))
        self.decoder = Sequential(
            Linear(latent, 256),
            ReLU(True),
            Linear(256, 256),
            ReLU(True), Linear(256, data_dim))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

class ResidualA(Module):
    def __init__(self, i, o):
        super(ResidualA, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)

class Decoder(Module):
    def __init__(self, embedding_dim, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        gen_dims = (64, 128)
        seq = []
        for item in list(gen_dims):
            seq += [
                ResidualA(dim, item)
            ]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return torch.tanh(data)

class AutoencoderRes(Module):
    def __init__(self, data_dim, latent):
        super(AutoencoderRes, self).__init__()
        self.encoder = Sequential(
            Linear(data_dim, 128),
            ReLU(True),
            Linear(128, 64),
            ReLU(True), Linear(64, latent))
        self.decoder = Decoder(latent, data_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat



class AEdis(Module):
    def __init__(self, input_dim, dis_dims, latent=20):
        super(AEdis, self).__init__()
        self.autoencoder = Autoencoder(input_dim, latent)
        seq = []
        self.dim = latent
        dim = self.dim
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        # input must be masked before fitting
        # mask_input = torch.mm(input, mask)
        input_ten = torch.unsqueeze(input, dim=1)
        z, x_hat = self.autoencoder(input_ten)
        x_hat = torch.sum(x_hat, dim=1)
        x = self.seq(z.view(-1, self.dim))
        return x, z, x_hat

class AEdisRes(Module):
    def __init__(self, input_dim, dis_dims, latent=20):
        super(AEdisRes, self).__init__()
        self.autoencoder = AutoencoderRes(input_dim, latent)
        seq = []
        self.dim = latent
        dim = self.dim
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        # input must be masked before fitting
        # mask_input = torch.mm(input, mask)
        input_ten = input
        z, x_hat = self.autoencoder(input_ten)
        x_hat = x_hat
        x = self.seq(z.view(-1, self.dim))
        return x, z, x_hat

class LAEdis(Module):
    def __init__(self, input_dim, dis_dims, latent=128):
        super(LAEdis, self).__init__()
        self.autoencoder = LAutoencoder(input_dim, latent)
        seq = []
        self.dim = latent
        dim = self.dim
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        # input must be masked before fitting
        # mask_input = torch.mm(input, mask)
        input_ten = torch.unsqueeze(input, dim=1)
        z, x_hat = self.autoencoder(input_ten)
        x_hat = torch.sum(x_hat, dim=1)
        x = self.seq(z.view(-1, self.dim))
        return x, z, x_hat