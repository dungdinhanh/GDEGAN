from models.ctgan import CTGANSynthesizer, Generator, Discriminator, apply_activate, calc_gradient_penalty, \
    cond_loss, Cond, Sampler, random_choice_prob_index
from models import configure
import torch
import torch.optim as optim
import os
import numpy as np


# this graph ctgan with conditional two D
class GCond(Cond):
    def __init__(self, data, output_info):
        # self.n_col = self.n_opt = 0
        # return
        super(GCond, self).__init__(data, output_info)

    def get_2_indexes(self, batch):
        idx = np.random.choice(np.arange(self.n_col), batch)
        idx2 = []
        for i in range(batch):
            idx2_item = np.random.choice(np.array(self.n_col), 1)[0]
            while idx2_item == idx[i]:
                idx2_item = np.random.choice(np.array(self.n_col), 1)[0]
            idx2.append(idx2_item)
        idx2 = np.array(idx2)
        return idx, idx2

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
        idx, idx2 = self.get_2_indexes(batch)

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        mask1[np.arange(batch), idx2] = 1
        opt1prime = random_choice_prob_index(self.p[idx])
        opt2prime = random_choice_prob_index(self.p[idx2])
        opt1 = self.interval[idx, 0] + opt1prime
        opt2 = self.interval[idx2, 0] + opt2prime
        vec1[np.arange(batch), opt1] = 1
        vec1[np.arange(batch), opt2] = 1

        return vec1, mask1, idx, opt1prime, idx2, opt2prime

    def sample_zero(self, batch):
        if self.n_col == 0:
            return None
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        # idx = np.random.choice(np.arange(self.n_col), batch)
        idx, idx2 = self.get_2_indexes(batch)
        for i in range(batch):
            col = idx[i]
            col2 = idx2[i]
            pick = int(np.random.choice(self.model[col]))
            pick2 = int(np.random.choice(self.model[col2]))
            vec[i, pick + self.interval[col, 0]] = 1
            vec[i, pick2 + self.interval[col2, 0]] = 1
        return vec

class GSampler(Sampler):
    """docstring for Sampler."""

    def __init__(self, data, output_info):
        super().__init__(data, output_info)

    def sample2cond(self, n, col, opt, col2, opt2):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for i in range(n):
            c = col[i]
            o = opt[i]
            c2 = col2[i]
            o2 = opt2[i]
            list_idx = np.array(self.model[c][o])
            list_idx2 = np.array(self.model[c2][o2])
            list_intersect = np.intersect1d(list_idx, list_idx2)
            if list_intersect.shape[0] == 0:
                list_intersect = np.union1d(list_idx, list_idx2)
            idx.append(np.random.choice(list_intersect))
        # for c, o in zip(col, opt):
        #     list_idx1 = np.array(self.model[c][o])
        #     idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]



class GraphCondCTGAN(CTGANSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self, embedding_dim=configure.EMBED_DIM, gen_dim=configure.GEN_DIM, dis_dim=configure.DIS_DIM,
                 l2scale=configure.WEIGHT_DECAY, batch_size=configure.BATCH, epochs=configure.EPOCHS,
                 cuda=configure.CUDA):
        super(GraphCondCTGAN, self).__init__(embedding_dim=embedding_dim, gen_dim=gen_dim, dis_dim=dis_dim,
                 l2scale=l2scale, batch_size=batch_size, epochs=epochs,
                 cuda=cuda)

        self.graph = configure.GRAPH

    def fit(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):

        self.transformer = configure.TRANSFORMER()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = GSampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dim
        self.cond_generator = GCond(train_data, self.transformer.output_info)

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
                    c1, m1, col, opt, col2, opt2 = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample2cond(self.batch_size, col[perm], opt[perm], col2[perm], opt2[perm])
                    # real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
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
                    c1, m1, col, opt, col2, opt2 = condvec
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
            print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
                            (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()))


    def fit_ckpt(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):

        self.transformer = configure.TRANSFORMER()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = GSampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dim
        self.cond_generator = GCond(train_data, self.transformer.output_info)

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
                    c1, m1, col, opt, col2, opt2 = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample2cond(self.batch_size, col[perm], opt[perm], col2[perm], opt2[perm])
                    # real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
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
                    c1, m1, col, opt, col2, opt2 = condvec
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
            print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
                            (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()))

            if (i + 1) >= 250 and (i + 1) % 10 == 0:
                ckpt_file = os.path.join(self.checkpoints_nets_folder, "epoch_%d.ckpt" % i)
                torch.save({
                    "generator": self.generator.state_dict(),
                    "transformer": self.transformer,
                    "cond_generator": self.cond_generator,
                    "n": train_data.shape[0],
                }, ckpt_file)
                self.generator.train()



