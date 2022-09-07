from sdgym.synthesizers import TableganSynthesizer, VEEGANSynthesizer
from sdgym.synthesizers.base import BaseSynthesizer, LOGGER
from utils.utils import get_columns_names
import os
from models import configure
import pandas as pd


class VEEGAN(VEEGANSynthesizer):

    def __init__(self):
        super(VEEGAN, self).__init__(embedding_dim=32,
        gen_dim=(128, 128),
        dis_dim=(128, ),
        rec_dim=(128, 128),
        l2scale=1e-6,
        batch_size=500,
        epochs=300)

        self.columns = configure.COLUMNS
        self.output_folder = configure.OUTPUT_FOLDER

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



        