import os
import pandas as pd
import torch.utils.data
from models import configure
from utils.utils import get_columns_names
from models.autoencoder_dis import *

CONTINUOUS = "continuous"
CATEGORICAL = "categorical"
ORDINAL = "ordinal"

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
from torch.nn import functional as F

from sdgym.synthesizers.base import BaseSynthesizer, LOGGER
from sdgym.synthesizers.utils import BGMTransformer


class Discriminator(Module):
    def __init__(self, input_dim, dis_dims, pack=configure.PAC):
        super(Discriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(dis_dims):
            seq += [
                Linear(dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            dim = item
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))


class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [
                Residual(dim, item)
            ]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)


def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


class Cond(object):
    def __init__(self, data, output_info):
        # self.n_col = self.n_opt = 0
        # return
        self.model = []

        st = 0
        skip = False
        max_interval = 0
        counter = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True
                continue
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                max_interval = max(max_interval, ed - st)
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed
            else:
                assert 0
        assert st == data.shape[1]

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        skip = False
        st = 0
        self.p = np.zeros((counter, max_interval))
        for item in output_info:
            if item[1] == 'tanh':
                skip = True
                st += item[0]
                continue
            elif item[1] == 'softmax':
                if skip:
                    st += item[0]
                    skip = False
                    continue
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0)
                tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                self.p[self.n_col, :item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed
            else:
                assert 0
        self.interval = np.asarray(self.interval)

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        opt1prime = random_choice_prob_index(self.p[idx])
        opt1 = self.interval[idx, 0] + opt1prime
        vec1[np.arange(batch), opt1] = 1

        return vec1, mask1, idx, opt1prime

    def sample_zero(self, batch):
        if self.n_col == 0:
            return None
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1
        return vec


def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    skip = False
    for item in output_info:
        if item[1] == 'tanh':
            st += item[0]
            skip = True

        elif item[1] == 'softmax':
            if skip:
                skip = False
                st += item[0]
                continue

            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
                data[:, st:ed],
                torch.argmax(c[:, st_c:ed_c], dim=1),
                reduction='none'
            )
            loss.append(tmp)
            st = ed
            st_c = ed_c

        else:
            assert 0
    loss = torch.stack(loss, dim=1)

    return (loss * m).sum() / data.size()[0]


class Sampler(object):
    """docstring for Sampler."""

    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)

        st = 0
        skip = False
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed
            else:
                assert 0
        assert st == data.shape[1]

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]


def calc_gradient_penalty(netD, real_data, fake_data, device='cpu', pac=configure.PAC, lambda_=10):
    # print("pac: %d"%pac)
    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    # interpolates = torch.Variable(interpolates, requires_grad=True, device=device)

    disc_interpolates,_, _ = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (
                               (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty

def calc_gradient_penalty2(netD, netA, real_data, fake_data, device='cpu', pac=configure.PAC, lambda_=10, latent=20):
    # print("pac: %d"%pac)
    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    # interpolates = torch.Variable(interpolates, requires_grad=True, device=device)
    z_f, x_hat_f = netA(interpolates)
    # x_hat_f = torch.sum(x_hat_f, dim=1)
    z_f = z_f.view(-1, latent)
    disc_interpolates = netD(z_f)

    # disc_interpolates,_, _ = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (
                               (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty


class CTGANSynthesizer(BaseSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self, embedding_dim=configure.EMBED_DIM, gen_dim=configure.GEN_DIM, dis_dim=configure.DIS_DIM,
                 l2scale=configure.WEIGHT_DECAY, batch_size=configure.BATCH, epochs=configure.EPOCHS,
                 cuda=configure.CUDA ):
        print("batchsize: %d" % batch_size)
        print("epoch: %d" % epochs)
        print("pac: %d"%configure.PAC)
        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent = configure.LATENT
        self.alpha = configure.ALPHA
        print("latent: %d" % self.latent)
        print("alpha: %f" % self.alpha)
        self.device = torch.device("cuda:%d" % cuda if torch.cuda.is_available() else "cpu")

        #folder and logs
        self.output_folder = configure.OUTPUT_FOLDER
        self.checkpoints = os.path.join(self.output_folder, "checkpoints")
        os.makedirs(self.checkpoints, exist_ok=True)

        self.checkpoints_nets_folder = os.path.join(self.checkpoints, "ckpt_nets")
        self.checkpoints_files_folder = os.path.join(self.checkpoints, "ckpt_files")
        os.makedirs(self.checkpoints_files_folder, exist_ok=True)
        os.makedirs(self.checkpoints_nets_folder, exist_ok=True)

        self.columns = configure.COLUMNS

    def fit(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):

        self.transformer = configure.TRANSFORMER()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)

        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim).to(self.device)

        discriminator = Discriminator(
            self.latent,
            self.dis_dim, 1).to(self.device)
        # discriminator = AEdis(data_dim, self.dis_dim, latent=self.latent).to(self.device)

        auto_encoder = Autoencoder(data_dim, self.latent).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        optimizerA = optim.Adam(auto_encoder.parameters(), lr=2e-4, betas=(0.5, 0.9))

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1
        steps_per_epoch = len(train_data) // self.batch_size
        for i in range(self.epochs):
            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)

                real = torch.from_numpy(real.astype('float32')).to(self.device)

                # if c1 is not None:
                #     fake_cat = torch.cat([fakeact, c1], dim=1)
                #     real_cat = torch.cat([real, c2], dim=1)
                # else:
                # real_cat = torch.unsqueeze(real, dim=1)
                # fake_cat = torch.unsqueeze(fakeact, dim=1)
                # fake_cat = fakeact
                real_cat = real
                fake_cat = fakeact

                realin = torch.unsqueeze(real_cat, dim=1)
                fakein = torch.unsqueeze(fake_cat, dim=1)


                z_f, x_hat_f = auto_encoder(fakein)
                x_hat_f = torch.sum(x_hat_f, dim=1)
                z_f = z_f.view(-1, self.latent)
                y_fake = discriminator(z_f)

                z_r, x_hat_r = auto_encoder(realin)
                x_hat_r = torch.sum(x_hat_r, dim=1)
                z_r = z_r.view(-1, self.latent)
                y_real = discriminator(z_r)

                # loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                m_out_real = torch.mean(y_real)
                m_out_fake = torch.mean(y_fake)
                loss_d = -(m_out_real - m_out_fake)

                # pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, self.device)
                pen = calc_gradient_penalty2(discriminator, auto_encoder, real_cat, fake_cat, self.device,
                                             latent=self.latent)
                recon_loss = self.alpha * (F.mse_loss(x_hat_f, fake_cat) + F.mse_loss(x_hat_r, real_cat))

                optimizerD.zero_grad()
                optimizerA.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward(retain_graph=True)
                recon_loss.backward()
                optimizerD.step()
                optimizerA.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)

                # if c1 is not None:
                #     y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                # else:
                # y_fake, z_f, x_hat_f = discriminator(fakeact)
                z_f, x_hat_f = auto_encoder(fakeact)
                x_hat_f = torch.sum(x_hat_f, dim=1)
                z_f = z_f.view(-1, self.latent)
                y_fake = discriminator(z_f)


                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy
                # if count_iter % 1000 == 0:
                #     writer.add_scalar("Loss/loss_g", loss_g, count_iter)
                #     writer.add_scalar("Loss/entropy_g", cross_entropy, count_iter)

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
            print("Epoch %d, Loss G: %.4f (loss entropy: %.4f), Loss D: %.4f (out real: %.4f, out fake: %.4f), reconstruct: %.4f" %
                (i + 1, loss_g.detach().cpu(), cross_entropy.detach().cpu(), loss_d.detach().cpu(),
                 m_out_real.detach().cpu(), m_out_fake.detach().cpu(), recon_loss.detach().cpu()))
            # print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
            #                 (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()))

    def fit_ckpt(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):

        self.transformer = configure.TRANSFORMER()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)

        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim).to(self.device)

        discriminator = Discriminator(
            self.latent,
            self.dis_dim, 1).to(self.device)
        # discriminator = AEdis(data_dim, self.dis_dim, latent=self.latent).to(self.device)

        auto_encoder = Autoencoder(data_dim, self.latent).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        optimizerA = optim.Adam(auto_encoder.parameters(), lr=2e-4, betas=(0.5, 0.9))

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1
        steps_per_epoch = len(train_data) // self.batch_size
        for i in range(self.epochs):
            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)

                real = torch.from_numpy(real.astype('float32')).to(self.device)

                # if c1 is not None:
                #     fake_cat = torch.cat([fakeact, c1], dim=1)
                #     real_cat = torch.cat([real, c2], dim=1)
                # else:
                real_cat = real
                fake_cat = fakeact

                real_cat = torch.unsqueeze(real_cat, dim=1)
                fake_cat = torch.unsqueeze(fake_cat, dim=1)
                # fake_cat = fakeact

                z_f, x_hat_f = auto_encoder(fake_cat)
                x_hat_f = torch.sum(x_hat_f, dim=1)
                z_f = z_f.view(-1, self.latent)
                y_fake = discriminator(z_f)

                z_r, x_hat_r = auto_encoder(real_cat)
                x_hat_r = torch.sum(x_hat_r, dim=1)
                z_r = z_r.view(-1, self.latent)
                y_real = discriminator(z_r)

                # loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                m_out_real = torch.mean(y_real)
                m_out_fake = torch.mean(y_fake)
                loss_d = -(m_out_real - m_out_fake)

                pen = calc_gradient_penalty2(discriminator, auto_encoder ,real_cat, fake_cat, self.device, latent=self.latent)
                recon_loss = self.alpha * (F.mse_loss(x_hat_f, fake_cat) + F.mse_loss(x_hat_r, real_cat))

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward(retain_graph=True)
                recon_loss.backward()
                optimizerD.step()
                optimizerA.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)

                # if c1 is not None:
                #     y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                # else:
                # y_fake, z_f, x_hat_f = discriminator(fakeact)
                z_f, x_hat_f = auto_encoder(fakeact)
                x_hat_f = torch.sum(x_hat_f, dim=1)
                z_f = z_f.view(-1, self.latent)
                y_fake = discriminator(z_f)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy
                # if count_iter % 1000 == 0:
                #     writer.add_scalar("Loss/loss_g", loss_g, count_iter)
                #     writer.add_scalar("Loss/entropy_g", cross_entropy, count_iter)

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
            print(
                "Epoch %d, Loss G: %.4f (loss entropy: %.4f), Loss D: %.4f (out real: %.4f, out fake: %.4f), reconstruct: %.4f" %
                (i + 1, loss_g.detach().cpu(), cross_entropy.detach().cpu(), loss_d.detach().cpu(),
                 m_out_real.detach().cpu(), m_out_fake.detach().cpu(), recon_loss.detach().cpu()))

            # save checkpoint
            if (i + 1) >= 200 and (i + 1) % 10 == 0:
                ckpt_file = os.path.join(self.checkpoints_nets_folder, "epoch_%d.ckpt" % i)
                torch.save({
                    "generator": self.generator.state_dict(),
                    "transformer": self.transformer,
                    "cond_generator": self.cond_generator,
                    "n": train_data.shape[0],
                }, ckpt_file)
                self.generator.train()

            # print("Epoch %d, Loss G: %.4f (loss entropy: %.4f), Loss D: %.4f (out real: %.4f, out fake: %.4f), reconstruct: %.4f" %
            #     (i + 1, loss_g.detach().cpu(), cross_entropy.detach().cpu(), loss_d.detach().cpu(),
            #      m_out_real.detach().cpu(), m_out_fake.detach().cpu(), recon_loss.detach().cpu()))


    def sample(self, n):
        self.generator.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.cond_generator.sample_zero(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = apply_activate(fake, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data, None)

    def sample_checkpoints(self, columns):
        for i in range(self.epochs):
            if (i + 1) >= 200 and (i + 1) % 10 == 0:
                ckpt_file = os.path.join(self.checkpoints_nets_folder, "epoch_%d.ckpt" % i)
                checkpoint = torch.load(ckpt_file)
                self.transformer = checkpoint['transformer']
                self.cond_generator = checkpoint['cond_generator']
                self.generator = Generator(
                            self.embedding_dim + self.cond_generator.n_opt,
                            self.gen_dim, self.transformer.output_dim).to(self.device)

                self.generator.load_state_dict(checkpoint['generator'])


                output_ckpt_file_folder = self.checkpoints_files_folder
                os.makedirs(output_ckpt_file_folder, exist_ok=True)
                output_ckpt_file = os.path.join(output_ckpt_file_folder, "epoch_%d.csv"%i)
                synth = self.sample(checkpoint['n'])
                synth = pd.DataFrame(synth, columns=columns)
                synth.to_csv(output_ckpt_file, index=False)
            pass

    def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit(data, categorical_columns, ordinal_columns)

        LOGGER.info("Sampling %s", self.__class__.__name__)
        data = self.sample(data.shape[0])
        if self.columns is None:
            dtframe = pd.DataFrame(data, columns=get_columns_names(data.shape[1]))
        else:
            dtframe = pd.DataFrame(data, columns=self.columns)
        # get file path and check if it exist
        output_file = os.path.join(self.output_folder, "output_0.csv")
        count_diff=1
        while os.path.isfile(output_file):
            output_file = os.path.join(self.output_folder, "output_%d.csv"%count_diff)
            count_diff += 1
        dtframe.to_csv(output_file, index=False)
        return data

    def fit_sample_ret(self, data, categorical_columns=tuple(), ordinal_columns=tuple(), sample_num=None):
        if sample_num is None:
            sample_num = data.shape[0]
        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit_ckpt(data, categorical_columns, ordinal_columns)

        LOGGER.info("Sampling %s", self.__class__.__name__)
        data = self.sample(sample_num)
        # if self.columns is None:
        #     dtframe = pd.DataFrame(data, columns=get_columns_names(data.shape[1]))
        # else:
        #     dtframe = pd.DataFrame(data, columns=self.columns)
        # # get file path and check if it exist
        # output_file = os.path.join(self.output_folder, "output_0.csv")
        # count_diff=1
        # while os.path.isfile(output_file):
        #     output_file = os.path.join(self.output_folder, "output_%d.csv"%count_diff)
        #     count_diff += 1
        # # dtframe.to_csv(output_file, index=False)
        return data

# class CTGANSynthesizer(BaseSynthesizer):
#     """docstring for IdentitySynthesizer."""
#
#     def __init__(self, embedding_dim=configure.EMBED_DIM, gen_dim=configure.GEN_DIM, dis_dim=configure.DIS_DIM,
#                  l2scale=configure.WEIGHT_DECAY, batch_size=configure.BATCH, epochs=configure.EPOCHS,
#                  cuda=configure.CUDA, output=configure.OUTPUT_FOLDER):
#
#         print("batchsize: %d"%batch_size)
#         print("epoch: %d"%epochs)
#         # parameters set up
#         self.embedding_dim = embedding_dim
#         self.gen_dim = gen_dim
#         self.dis_dim = dis_dim
#
#         self.l2scale = l2scale
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.exp_pac_size = 1
#         # self.now = datetime.datetime.now().time()
#
#         self.columns = None
#
#         # folder and logs
#         self.output_folder = output
#         self.checkpoints = os.path.join(self.output_folder, "checkpoints")
#         os.makedirs(self.checkpoints, exist_ok=True)
#
#         if torch.cuda.is_available():
#             self.device = torch.device("cuda:"+str(cuda))
#         else:
#             self.device=None
#             print("No GPUs!")
#             exit()
#
#     def fit(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):
#
#         # transfomer data
#         self.transformer = sdgymutils.BGMTransformer()
#         self.transformer.fit(train_data, categorical_columns, ordinal_columns)
#         train_data = self.transformer.transform(train_data)
#
#
#         # Data sampler for conditional vectors
#         data_sampler = Sampler(train_data, self.transformer.output_info)
#
#         data_dim = self.transformer.output_dim
#         self.cond_generator = Cond(train_data, self.transformer.output_info)
#
#         self.generator = Generator(
#             self.embedding_dim + self.cond_generator.n_opt,
#             self.gen_dim,
#             data_dim
#         ).to(self.device)
#
#         discriminator = Discriminator(
#             data_dim + self.cond_generator.n_opt,
#             self.dis_dim
#         ).to(self.device)
#
#         optimizerG = optim.Adam(
#             self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9),
#             weight_decay=self.l2scale
#         )
#         optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
#
#         assert self.batch_size % 2 == 0
#         mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
#         std = mean + 1
#
#         steps_per_epoch = len(train_data) // self.batch_size
#         for i in range(self.epochs):
#             for id_ in range(steps_per_epoch):
#                 fakez = torch.normal(mean=mean, std=std)
#
#                 condvec = self.cond_generator.sample(self.batch_size)
#                 if condvec is None:
#                     c1, m1, col, opt = None, None, None, None
#                     real = data_sampler.sample(self.batch_size, col, opt)
#                 else:
#                     c1, m1, col, opt = condvec
#                     c1 = torch.from_numpy(c1).to(self.device)
#                     m1 = torch.from_numpy(m1).to(self.device)
#                     fakez = torch.cat([fakez, c1], dim=1)
#
#                     perm = np.arange(self.batch_size)
#                     np.random.shuffle(perm)
#                     real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
#                     c2 = c1[perm]
#
#                 fake = self.generator(fakez)
#                 fakeact = apply_activate(fake, self.transformer.output_info)
#
#                 real = torch.from_numpy(real.astype('float32')).to(self.device)
#
#                 if c1 is not None:
#                     fake_cat = torch.cat([fakeact, c1], dim=1)
#                     real_cat = torch.cat([real, c2], dim=1)
#                 else:
#                     real_cat = real
#                     fake_cat = fake
#
#                 y_fake = discriminator(fake_cat)
#                 y_real = discriminator(real_cat)
#
#                 loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
#                 pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, self.device)
#
#                 optimizerD.zero_grad()
#                 pen.backward(retain_graph=True)
#                 loss_d.backward()
#                 optimizerD.step()
#
#                 fakez = torch.normal(mean=mean, std=std)
#                 condvec = self.cond_generator.sample(self.batch_size)
#
#                 if condvec is None:
#                     c1, m1, col, opt = None, None, None, None
#                 else:
#                     c1, m1, col, opt = condvec
#                     c1 = torch.from_numpy(c1).to(self.device)
#                     m1 = torch.from_numpy(m1).to(self.device)
#                     fakez = torch.cat([fakez, c1], dim=1)
#
#                 fake = self.generator(fakez)
#                 fakeact = apply_activate(fake, self.transformer.output_info)
#
#                 if c1 is not None:
#                     y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
#                 else:
#                     y_fake = discriminator(fakeact)
#
#                 if condvec is None:
#                     cross_entropy = 0
#                 else:
#                     cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)
#
#                 loss_g = -torch.mean(y_fake) + cross_entropy
#
#                 optimizerG.zero_grad()
#                 loss_g.backward()
#                 optimizerG.step()
#
#                 # if i > 200:
#                 #     if (i+1) % 10 == 0:
#                 #         folder_epoch = os.path.join(self.log_epoch_folder, "epoch%d%s"%(i, self.output_logepoch_file))
#                 #         epoch_samples = self.sample(self.epoch_sample_num)
#                 #         write_epoch_data(epoch_samples, self.columns_name, None, folder_epoch)
#
#             print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
#                   (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()))
#             # torch.save(self.generator.state_dict(), self.model_path)
#
#     def sample(self, n):
#
#         output_info = self.transformer.output_info
#         steps = n // self.batch_size + 1
#         data = []
#         for i in range(steps):
#             mean = torch.zeros(self.batch_size, self.embedding_dim)
#             std = mean + 1
#             fakez = torch.normal(mean=mean, std=std).to(self.device)
#
#             condvec = self.cond_generator.sample_zero(self.batch_size)
#             if condvec is None:
#                 pass
#             else:
#                 c1 = condvec
#                 c1 = torch.from_numpy(c1).to(self.device)
#                 fakez = torch.cat([fakez, c1], dim=1)
#
#             fake = self.generator(fakez)
#             fakeact = apply_activate(fake, output_info)
#             data.append(fakeact.detach().cpu().numpy())
#
#         data = np.concatenate(data, axis=0)
#         data = data[:n]
#         return self.transformer.inverse_transform(data, None)
#
#     def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
#         LOGGER.info("Fitting %s", self.__class__.__name__)
#         self.fit(data, categorical_columns, ordinal_columns)
#
#         LOGGER.info("Sampling %s", self.__class__.__name__)
#         data = self.sample(data.shape[0])
#         if self.columns is None:
#             dtframe = pd.DataFrame(data, columns=get_columns_names(data.shape[1]))
#             output_file = os.path.join(self.output_folder, "output.csv")
#             dtframe.to_csv(output_file, index=False)
#         return data
#
#     # def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
#     #     LOGGER.info("Fitting %s", self.__class__.__name__)
#     #     self.fit(data, categorical_columns, ordinal_columns)
#     #
#     #     LOGGER.info("Sampling %s", self.__class__.__name__)
#     #     return self.sample(data.shape[0])
