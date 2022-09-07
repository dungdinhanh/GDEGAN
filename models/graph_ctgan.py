from models.ctgan import CTGANSynthesizer, Generator, Discriminator, apply_activate, calc_gradient_penalty, \
    cond_loss, Sampler, random_choice_prob_index
from models import configure
import torch
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import joblib


class GCond(object):
    def __init__(self, data, output_info, graph):
        self.model = []
        self.data = data
        self.graph = graph
        # self.cat2column_dict = {}
        # self.column2cat_dict = {}
        self.con_column = []

        st = 0
        skip = False
        max_interval = 0
        counter = 0
        column_counter = 0
        con_counter = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True
                self.con_column.append(column_counter)
                con_counter += 1
                column_counter += 1
                continue
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                max_interval = max(max_interval, ed - st)
                # self.cat2column_dict[counter] = column_counter
                # self.column2cat_dict[column_counter] = counter
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed
            else:
                assert 0
            column_counter += 1
        assert st == data.shape[1]
        self.con_column = np.asarray(self.con_column).astype(int)

        self.column2cat = np.full(column_counter, -1)
        self.cat2column = np.full(counter, -1)
        cat_counter = 0
        column_counter = 0
        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        skip = False
        st = 0
        self.p = np.zeros((counter, max_interval))
        for index, item in enumerate(output_info):
            if item[1] == 'tanh':
                skip = True
                st += item[0]
                column_counter += 1
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

                self.column2cat[column_counter] = cat_counter
                self.cat2column[cat_counter] = column_counter

                self.n_opt += item[0]
                self.n_col += 1
                st = ed
                cat_counter += 1
            else:
                assert 0
            column_counter += 1
        self.interval = np.asarray(self.interval)
        self.model = np.asarray(self.model, dtype="int")

        # process graph: # for categorical chosen
        if self.graph is not None:
            nodes = self.graph.shape[0]
            self.graph[np.arange(nodes), np.arange(nodes)] = 1
            if self.con_column.shape[0] > 0:
                self.graph[:, self.con_column] = 0
                self.graph[self.con_column, :] = 0

    def get_2_indexes(self, batch):
        if self.graph is None:
            idx = np.random.choice(np.arange(self.n_col), batch)
            idx2 = []
            for i in range(batch):
                idx2_item = np.random.choice(np.array(self.n_col), 1)[0]
                while idx2_item == idx[i]:
                    idx2_item = np.random.choice(np.array(self.n_col), 1)[0]
                idx2.append(idx2_item)
            idx2 = np.array(idx2)
        else:
            nonzeros_el = self.graph.nonzero()
            num = nonzeros_el[0].shape[0]
            selected = np.random.choice(np.arange(num), batch)
            column_idx = nonzeros_el[0][selected]
            column_idx2 = nonzeros_el[1][selected]
            idx = self.column2cat[column_idx]
            idx2 = self.column2cat[column_idx2]
        return [idx, idx2]

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
        indexes = self.get_2_indexes(batch)
        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        real_data_indexes = np.random.choice(np.arange(len(self.model[0])), batch)
        for idx in indexes:
            mask1[np.arange(batch), idx] = 1
            opt1 = self.interval[idx, 0] + self.model[idx, real_data_indexes]
            opt1 = opt1.astype(int)
            vec1[np.arange(batch), opt1] = 1
        return vec1, mask1, indexes, self.data[real_data_indexes]

    def sample_temp(self, batch):
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
        batch = batch
        indexes = self.get_2_indexes(batch)
        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        real_data_indexes = np.random.choice(np.arange(len(self.model[0])), batch)
        for idx in indexes:
            mask1[np.arange(batch), idx] = 1
            opt1 = self.interval[idx, 0] + self.model[idx, real_data_indexes]
            opt1 = opt1.astype(int)
            vec1[np.arange(batch), opt1] = 1
        return vec1

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
        self.cond_generator = GCond(train_data, self.transformer.output_info, self.graph)

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
                    c1, m1, cols, real= condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = real[perm]
                    # real = data_sampler.sample2cond(self.batch_size, col[perm], opt[perm], col2[perm], opt2[perm])
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
                    c1, m1, cols, real = condvec
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
        self.cond_generator = GCond(train_data, self.transformer.output_info, self.graph)

        trans_cond_dict = {
            "transformer": self.transformer,
            "cond_generator": self.cond_generator,
        }
        # torch.save(trans_cond_dict,  os.path.join(self.checkpoints_nets_folder, "trans_cond.ckpt"))
        joblib.dump(trans_cond_dict, os.path.join(self.checkpoints_nets_folder, "trans_cond.ckpt"))

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
                    c1, m1, cols, real = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = real[perm]
                    # real = data_sampler.sample2cond(self.batch_size, col[perm], opt[perm], col2[perm], opt2[perm])
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
                    c1, m1, cols, real = condvec
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
                    "n": train_data.shape[0],
                }, ckpt_file)
                self.generator.train()

    def sample_checkpoints(self, columns, old_version=False):
        if not old_version:
            self.new_sample_checkpoints(columns)
        else:
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


    def new_sample_checkpoints(self, columns):
        trans_cond_file = os.path.join(self.checkpoints_nets_folder, "trans_cond.ckpt")
        if not os.path.isfile(trans_cond_file):
            return self.sample_checkpoints(columns=columns, old_version=True)
        trans_cond_dict = joblib.load(trans_cond_file)
        self.transformer = trans_cond_dict['transformer']
        self.cond_generator = trans_cond_dict['cond_generator']
        for i in range(self.epochs):
            if (i + 1) >= 250 and (i + 1) % 10 == 0:
            # if (i + 1) >= 0 and (i + 1) % 5 == 0:
                ckpt_file = os.path.join(self.checkpoints_nets_folder, "epoch_%d.ckpt" % i)

                checkpoint = torch.load(ckpt_file)

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



