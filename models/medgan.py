import os
import glob
import pandas as pd
import torch.utils.data
from models import configure
from utils.utils import get_columns_names

CONTINUOUS = "continuous"
CATEGORICAL = "categorical"
ORDINAL = "ordinal"


from sdgym.synthesizers.base import BaseSynthesizer, LOGGER
from sdgym.synthesizers.utils import BGMTransformer
from sdgym.synthesizers import MedganSynthesizer

class MedGAN(MedganSynthesizer):
    def __init__(self):
        super(MedGAN, self).__init__(
                     embedding_dim=128,
                     random_dim=128,
                     generator_dims=(128, 128),  # 128 -> 128 -> 128
                     discriminator_dims=(256, 128, 1),  # datadim * 2 -> 256 -> 128 -> 1
                     compress_dims=(),  # datadim -> embedding_dim
                     decompress_dims=(),  # embedding_dim -> datadim
                     bn_decay=0.99,
                     l2scale=0.001,
                     pretrain_epoch=200,
                     batch_size=1000,
                     epochs=2000)

        self.columns = configure.COLUMNS
        self.output_folder = configure.OUTPUT_FOLDER
        os.makedirs(self.output_folder, exist_ok=True)


    def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit(data, categorical_columns, ordinal_columns)

        LOGGER.info("Sampling %s", self.__class__.__name__)
        data = self.sample(data.shape[0])

        if self.columns is None:
            dtframe = pd.DataFrame(data, columns=get_columns_names(data.shape[1]))
        else:
            dtframe = pd.DataFrame(data, columns=self.columns)

        self.clear_cache()
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

    def clear_cache(self):
        files = glob.glob(os.path.join(self.output_folder, "*.csv"))
        for f in files:
            os.remove(f)
        pass






