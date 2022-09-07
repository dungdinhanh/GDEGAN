from models.daegan import calc_gradient_penalty, Generator, Sampler, apply_activate, cond_loss
from models import configure
import torch
import torch.optim as optim
import os
import numpy as np
from models.graph_ctgan import GraphCondCTGAN, GSampler, GCond
from models.autoencoder_dis import LAEdis, LAutoencoder
from torch.nn import functional as F
import joblib




class GraphCondLargeDAEGAN(GraphCondCTGAN):
    """docstring for IdentitySynthesizer."""

    def __init__(self, embedding_dim=configure.EMBED_DIM, gen_dim=configure.GEN_DIM, dis_dim=configure.DIS_DIM,
                 l2scale=configure.WEIGHT_DECAY, batch_size=configure.BATCH, epochs=configure.EPOCHS,
                 cuda=configure.CUDA):
        super(GraphCondLargeDAEGAN, self).__init__(embedding_dim=embedding_dim, gen_dim=gen_dim, dis_dim=dis_dim,
                 l2scale=l2scale, batch_size=batch_size, epochs=epochs,
                 cuda=cuda)

        # self.graph = configure.GRAPH
        self.latent = configure.LATENT
        self.alpha = configure.ALPHA
        print("latent: %d" % self.latent)
        print("alpha: %f" % self.alpha)

    def fit(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.transformer = configure.TRANSFORMER()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)

        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dim
        # self.cond_generator = Cond(train_data, self.transformer.output_info)
        self.cond_generator = GCond(train_data, self.transformer.output_info, self.graph)
        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim).to(self.device)

        # discriminator = Discriminator(
        #     data_dim + self.cond_generator.n_opt,
        #     self.dis_dim).to(self.device)
        discriminator = LAEdis(data_dim, self.dis_dim, latent=self.latent).to(self.device)

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

                # if c1 is not None:
                #     fake_cat = torch.cat([fakeact, c1], dim=1)
                #     real_cat = torch.cat([real, c2], dim=1)
                # else:
                real_cat = real
                fake_cat = fakeact

                y_fake, z_f, x_hat_f = discriminator(fake_cat)
                y_real, z_r, x_hat_r = discriminator(real_cat)

                # loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                m_out_real = torch.mean(y_real)
                m_out_fake = torch.mean(y_fake)
                loss_d = -(m_out_real - m_out_fake)

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
                    c1, m1, cols, real = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)
                y_fake, z_f, x_hat_f = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy


                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
            print(
                "Epoch %d, Loss G: %.4f , Loss D: %.4f (out real: %.4f, out fake: %.4f), reconstruct: %.4f" %
                (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu(),
                 m_out_real.detach().cpu(), m_out_fake.detach().cpu(), recon_loss.detach().cpu()))

    def fit_ckpt(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.transformer = configure.TRANSFORMER()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)

        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dim
        # self.cond_generator = Cond(train_data, self.transformer.output_info)
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

        # discriminator = Discriminator(
        #     data_dim + self.cond_generator.n_opt,
        #     self.dis_dim).to(self.device)
        discriminator = LAEdis(data_dim, self.dis_dim, latent=self.latent).to(self.device)

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

                # if c1 is not None:
                #     fake_cat = torch.cat([fakeact, c1], dim=1)
                #     real_cat = torch.cat([real, c2], dim=1)
                # else:
                real_cat = real
                fake_cat = fakeact

                y_fake, z_f, x_hat_f = discriminator(fake_cat)
                y_real, z_r, x_hat_r = discriminator(real_cat)

                # loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                m_out_real = torch.mean(y_real)
                m_out_fake = torch.mean(y_fake)
                loss_d = -(m_out_real - m_out_fake)

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
                    c1, m1, cols, real = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)
                y_fake, z_f, x_hat_f = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
            print(
                "Epoch %d, Loss G: %.4f , Loss D: %.4f (out real: %.4f, out fake: %.4f), reconstruct: %.4f" %
                (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu(),
                 m_out_real.detach().cpu(), m_out_fake.detach().cpu(), recon_loss.detach().cpu()))

            if (i + 1) >= 250 and (i + 1) % 10 == 0:
                ckpt_file = os.path.join(self.checkpoints_nets_folder, "epoch_%d.ckpt" % i)
                torch.save({
                    "generator": self.generator.state_dict(),
                    "n": train_data.shape[0],
                }, ckpt_file)
                self.generator.train()



