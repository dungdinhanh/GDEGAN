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
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
from torch.nn import functional as F

from sdgym.synthesizers.base import BaseSynthesizer, LOGGER
from sdgym.synthesizers.utils import BGMTransformer
from models.ctgan import CTGANSynthesizer
import joblib





class LCTGAN(CTGANSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self, embedding_dim=configure.EMBED_DIM, gen_dim=configure.GEN_DIM, dis_dim=configure.DIS_DIM,
                 l2scale=configure.WEIGHT_DECAY, batch_size=configure.BATCH, epochs=configure.EPOCHS,
                 cuda=configure.CUDA, temp=False):
        super(LCTGAN, self).__init__(embedding_dim=embedding_dim, gen_dim=gen_dim, dis_dim=dis_dim,
                                              l2scale=l2scale, batch_size=batch_size, epochs=epochs,
                                              cuda=cuda)
        self.dis_dim = (256, 256, 256, 256)


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


    # def get_2_indexes(self, batch):
    #     idx = np.random.choice(np.arange(self.n_col), batch)
    #     idx2 = np.random.choice(np.arange(self.n_col), batch)
    #     # for i in range(batch):
    #     #     idx2_item = np.random.choice(np.array(self.n_col), 1)[0]
    #     #     while idx2_item == idx[i]:
    #     #         idx2_item = np.random.choice(np.array(self.n_col), 1)[0]
    #     #     idx2.append(idx2_item)
    #     # idx2 = np.array(idx2)
    #     return idx, idx2