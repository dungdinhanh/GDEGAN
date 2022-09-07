import os
import glob
import pandas as pd
import torch.utils.data
from models import configure
from utils.utils import get_columns_names

CONTINUOUS = "continuous"
CATEGORICAL = "categorical"
ORDINAL = "ordinal"

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
from torch.nn import functional as F

from sdgym.synthesizers.base import BaseSynthesizer, LOGGER
from sdgym.synthesizers.utils import BGMTransformer
import matplotlib.pyplot as plt
from datetime import datetime
from models.autoencoder_dis import AEdis


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
    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    # interpolates = torch.Variable(interpolates, requires_grad=True, device=device)

    disc_interpolates, _, _ = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = (
                               (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty


class HD_DAECTGANSynthesizer(BaseSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self, embedding_dim=configure.EMBED_DIM, gen_dim=configure.GEN_DIM, dis_dim=configure.DIS_DIM,
                 l2scale=configure.WEIGHT_DECAY, batch_size=configure.BATCH, epochs=configure.EPOCHS,
                 cuda=configure.CUDA, ):
        print("batchsize: %d" % batch_size)
        print("epoch: %d" % epochs)
        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device("cuda:%d" % cuda if torch.cuda.is_available() else "cpu")

        # folder and logs
        self.output_folder = configure.OUTPUT_FOLDER
        self.checkpoints = os.path.join(self.output_folder, "checkpoints")
        os.makedirs(self.checkpoints, exist_ok=True)

        self.checkpoints_nets_folder = os.path.join(self.checkpoints, "ckpt_nets")
        self.checkpoints_files_folder = os.path.join(self.checkpoints, "ckpt_files")
        os.makedirs(self.checkpoints_files_folder, exist_ok=True)
        os.makedirs(self.checkpoints_nets_folder, exist_ok=True)

        self.columns = configure.COLUMNS
        self.num_one = configure.NUM_ONES
        self.now = datetime.now().time()
        self.compress_dims = [256, 256]
        self.encoder_embedding_dim = 128
        self.latent = configure.LATENT
        self.alpha = configure.ALPHA

    def adding_identity(self, train_data, num_one, categorical_columns):
        new_train_data = np.ones((train_data.shape[0], train_data.shape[1] * (num_one + 1)))
        new_categorical_columns = []
        indexes = []
        for i in range(train_data.shape[1]):
            st = i * (num_one + 1)
            new_train_data[:, st] = train_data[:, i]
            indexes.append(st)
            if (i in categorical_columns):
                new_categorical_columns.append(st)
            for j in range(st + 1, st + num_one + 1):
                new_categorical_columns.append(j)
        return new_train_data, new_categorical_columns, indexes

    def addingone(self, data):
        new_data = []
        for i in range(data.shape[1]):
            temp_data = data[:, i].reshape((data.shape[0], 1))
            new_data.append(temp_data)
            new_data.append(torch.ones((data.shape[0], self.num_one)).to(self.device))
        new_tensor = torch.cat(new_data, dim=1)
        return new_tensor

    def fit(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):
        # old_train_data = train_data.copy()
        # old_categorical_columns = categorical_columns.copy()
        if (self.num_one > 0):
            train_data, categorical_columns, self.indexes = self.adding_identity(train_data, self.num_one,
                                                                                 categorical_columns)
        else:
            self.indexes = range(train_data.shape[1])
            print(train_data.shape)

        self.transformer = BGMTransformer()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim).to(self.device)

        discriminator = AEdis(data_dim, self.dis_dim, latent=self.latent).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        # hd_distances = []
        # old_distances = []
        steps_per_epoch = len(train_data) // self.batch_size
        self.log_file = os.path.join(self.output_folder, "log.csv")
        f = open(self.log_file, "w")
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
                #     real_cat = real
                #     fake_cat = fake
                real_cat = real
                fake_cat = fakeact

                y_fake, _, x_hat_f = discriminator(fake_cat)
                y_real, _, x_hat_r = discriminator(real_cat)

                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, self.device)
                recon_loss = self.alpha * (F.mse_loss(x_hat_f, fake_cat) + F.mse_loss(x_hat_r, real_cat))

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward(retain_graph=True)
                recon_loss.backward()
                optimizerD.step()

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
                #     y_fake = discriminator(fakeact)
                y_fake, _, _ = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
                # if(id_==0):
                #     if(i>=0):
                #         hd_distances.append(-loss_d.item())
                #         old_distances.append(-old_loss_d.item())
                #         print("W distance with '1'", -loss_d.item(), "W distance no '1'", -old_loss_d.item())
                #         plt.plot(hd_distances, color = 'r', label="W distance with '1'")
                #         plt.plot(old_distances, color = 'g', label="W distance no '1'")
                #         if(i==0 and id_==0):
                #             plt.legend(loc="upper left")
                #         plt.savefig( os.path.join(self.output_folder, "WdistancevisualizationRealRealNoEncoder"+str(self.now)+".png"))
                f.write("%d, %.4f, %.4f\n" % (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()))
            print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
                  (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()))

        f.close()
        # plt.close()

    def fit_ckpt(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):

        self.transformer = BGMTransformer()
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
            data_dim + self.cond_generator.n_opt,
            self.dis_dim).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

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

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, self.device)

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()
                optimizerD.step()

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

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            # save checkpoint
            if (i + 1) >= 250 and (i + 1) % 10 == 0:
                ckpt_file = os.path.join(self.checkpoints_nets_folder, "epoch_%d.ckpt" % i)
                torch.save({
                    "generator": self.generator.state_dict(),
                    "transformer": self.transformer,
                    "cond_generator": self.cond_generator,
                    "n": train_data.shape[0],
                }, ckpt_file)
                self.generator.train()

            print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
                  (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()))

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
        return self.transformer.inverse_transform(data, None)[:, self.indexes]

    def sample_checkpoints(self, columns):
        for i in range(self.epochs):
            if (i + 1) >= 250 and (i + 1) % 10 == 0:
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
                output_ckpt_file = os.path.join(output_ckpt_file_folder, "epoch_%d.csv" % i)
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
        # self.clear_cache()
        # Start to store cache
        output_file = os.path.join(self.output_folder, "output_0.csv")
        count_diff = 1
        while os.path.isfile(output_file):
            output_file = os.path.join(self.output_folder, "output_%d.csv" % count_diff)
            count_diff += 1
        dtframe.to_csv(output_file, index=False)
        return data

    def fit_sample_ret(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit_ckpt(data, categorical_columns, ordinal_columns)

        LOGGER.info("Sampling %s", self.__class__.__name__)
        data = self.sample(data.shape[0])
        return data

    def clear_cache(self):
        files = glob.glob(os.path.join(self.output_folder, "*.csv"))
        for f in files:
            os.remove(f)
        pass