from sdgym.synthesizers.utils import BGMTransformer
from models.transformers import *

EPOCHS = 300
EMBED_DIM = 128
GEN_DIM = (256, 256)
DIS_DIM = (256, 256)
WEIGHT_DECAY = 1e-6
CUDA = 0
BATCH = 500
PAC = 10
OUTPUT_FOLDER="output"
COLUMNS=None


# TRANS configure
LIST_TRANSFORMER = [BGMTransformer, BGMTransformer2, BGMTransformer3]
TRANSFORMER = BGMTransformer

# DAE configure
LATENT=25
ALPHA=0.1

# Graph configure
GRAPH = None

# Priv Bayes configure
THETA = 20
MAX_SAMPLES = 25000


# High dimension
NUM_ONES = 0

# class Configure:
#     configure_data = None
#
#     def __init__(self):
#         if Configure.configure_data is not None:
#             print("Why calling this???? it is singleton")
#             exit(0)
#         Configure.configure_data = self
#         self.epoch = EPOCHS
#         self.embdim = EMBED_DIM
#         self.gen_dim = GEN_DIM
#         self.dis_dim = DIS_DIM
#         self.weight_decay = WEIGHT_DECAY
#         self.cuda = CUDA
#         self.batchs = BATCH
#         self.pac = PAC
#
#     @staticmethod
#     def get_configure():
#         if Configure.configure_data is None:
#             Configure.configure_data = Configure()
#         return Configure.configure_data



