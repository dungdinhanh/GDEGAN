from sdgym.synthesizers import TableganSynthesizer, VEEGANSynthesizer
from sdgym.synthesizers.veegan import Generator, Reconstructor, Discriminator, DataLoader, TensorDataset
from sdgym.synthesizers.base import BaseSynthesizer, LOGGER
from sdgym.synthesizers.utils import *
from torch.nn.functional import mse_loss, softmax
from torch.optim import Adam
import torch
from utils.utils import get_columns_names
import os
from models import configure
import pandas as pd
import numpy as np


class HDVEEGAN(VEEGANSynthesizer):

    def __init__(self):
        super(HDVEEGAN, self).__init__(embedding_dim=32,
        gen_dim=(128, 128),
        dis_dim=(128, ),
        rec_dim=(128, 128),
        l2scale=1e-6,
        batch_size=500,
        epochs=300)

        self.columns = configure.COLUMNS
        self.output_folder = configure.OUTPUT_FOLDER
        self.num_one = configure.NUM_ONES
        os.makedirs(self.output_folder, exist_ok=True)



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

    def fit(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):
        if (self.num_one > 0):
            train_data, categorical_columns, self.indexes = self.adding_identity(train_data, self.num_one,
                                                                                 categorical_columns)
        else:
            self.indexes = range(train_data.shape[1])
            print(train_data.shape)
        self.transformer = GeneralTransformer(act='tanh')
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dim
        self.generator = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
        discriminator = Discriminator(self.embedding_dim + data_dim, self.dis_dim).to(self.device)
        reconstructor = Reconstructor(data_dim, self.rec_dim, self.embedding_dim).to(self.device)

        optimizer_params = dict(lr=1e-3, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)
        optimizerR = Adam(reconstructor.parameters(), **optimizer_params)

        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1
        self.log_file = os.path.join(self.output_folder, "log.csv")
        f = open(self.log_file, "w")
        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                realz = reconstructor(real)
                y_real = discriminator(torch.cat([real, realz], dim=1))

                fakez = torch.normal(mean=mean, std=std)
                fake = self.generator(fakez, self.transformer.output_info)
                fakezrec = reconstructor(fake)
                y_fake = discriminator(torch.cat([fake, fakez], dim=1))

                loss_d = (
                    -(torch.log(torch.sigmoid(y_real) + 1e-4).mean())
                    - (torch.log(1. - torch.sigmoid(y_fake) + 1e-4).mean())
                )

                numerator = -y_fake.mean() + mse_loss(fakezrec, fakez, reduction='mean')
                loss_g = numerator / self.embedding_dim
                loss_r = numerator / self.embedding_dim
                optimizerD.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizerD.step()
                optimizerG.zero_grad()
                loss_g.backward(retain_graph=True)
                optimizerG.step()
                optimizerR.zero_grad()
                loss_r.backward()
                optimizerR.step()
                print("iter %d, Loss G: %.4f, Loss D: %.4f"%(i+1, loss_g.detach().cpu(), loss_d.detach().cpu()))
                f.write("%d, %.4f, %.4f\n"%(i+1, loss_g.detach().cpu(), loss_d.detach().cpu()))
        f.close()


    def sample(self, n):
        self.generator.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self.device)
            fake = self.generator(noise, output_info)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data)[:, self.indexes]

    def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit(data, categorical_columns, ordinal_columns)

        LOGGER.info("Sampling %s", self.__class__.__name__)
        data = self.sample(data.shape[0])

        if self.columns is None:
            dtframe = pd.DataFrame(data, columns=get_columns_names(data.shape[1]))
        else:
            dtframe = pd.DataFrame(data, columns=self.columns)


        output_file = os.path.join(self.output_folder, "output_0.csv")
        count_diff = 1
        while os.path.isfile(output_file):
            output_file = os.path.join(self.output_folder, "output_%d.csv"%count_diff)
            count_diff += 1
        dtframe.to_csv(output_file, index=False)

        return data

    def fit_sample_ret(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit(data, categorical_columns, ordinal_columns)

        LOGGER.info("Sampling %s", self.__class__.__name__)
        data = self.sample(data.shape[0])
        return data



        