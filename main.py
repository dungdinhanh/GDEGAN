import sdgym

import argparse
import os


from models import configure

# from sdgym.metrics import get_metrics
# from sdgym.datasets import get_dataset_paths, load_dataset, load_tables
# from sdmetrics.single_table.efficacy.binary import BinaryDecisionTreeClassifier
'''
DEFAULT_DATASETS = [
    "adult", #1
    "alarm", 
    "asia",
    "census", #2
    "child",
    "covtype", #3
    "credit", #4
    "grid",
    "gridr",
    "insurance",
    "intrusion", #5
    "mnist12",
    "mnist28", #6
    "news",
    "ring"
]
SSVEEGANSynthesizer: 12
DGICTGANSynthesizer: 20
'''

DEFAULT_DATASETS = [
    "adult", #1
    "alarm",
    "asia",
    "census", #2
    "child",
    "covtype", #3
    "credit", #4
    "grid",
    "gridr",
    "insurance",
    "intrusion", #5
    "mnist12",
    "mnist28", #6
    "news",
    "ring"
]

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', default=4 ,type=int, help='index of model')
parser.add_argument('--dataset', default="adult", type=str ,help='benchmark dataset')
parser.add_argument('--epochs', default=300, type=int, help='Number of epochs for training')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
parser.add_argument('--batch', default=500, type=int, help='batch size')
parser.add_argument('--outputdir', default='output', type=str, help='output folder')
parser.add_argument('--latent', default=25, type=int, help='latent size for autoencoder')
parser.add_argument('--alpha', default=0.1, type=float, help='weight for reconstruction loss')
parser.add_argument('--test', action='store_true', help='train or load checkpoints')
parser.add_argument('--transformer', default=0, type=int, help="index for choosing transformer, 0 for normal, 1 for discretize low card integer")
# parser = argparse.ArgumentParser(description='Process some integers.')
args = parser.parse_args()
configure.EPOCHS = args.epochs
configure.BATCH = args.batch
configure.CUDA = args.cuda
configure.TRANSFORMER = configure.LIST_TRANSFORMER[args.transformer]

# must be imported after changing configure (import before causing errors)
from models.ctgan import CTGANSynthesizer
from models.daegan import DEGANSynthesizer
from models.graph_daegan import GraphCondDAEGAN
from models.graph_ctgan import GraphCondCTGAN
from models.daegan_res import DEGANResSynthesizer

models = [CTGANSynthesizer, DEGANSynthesizer, DEGANResSynthesizer, GraphCondCTGAN, GraphCondDAEGAN]
from utils. utils import *
import json
from utils.dataset_stat import *


if __name__ == '__main__':
    dataset_name = args.dataset
    configure.LATENT = args.latent
    alpha = configure.ALPHA = args.alpha
    # Get folder mark
    model = models[args.model]

    configure.OUTPUT_FOLDER = os.path.join(configure.OUTPUT_FOLDER, model.__class__.__name__, args.dataset)
    if not args.test:
        # This folder is the tuning folders
        output_dir = configure.OUTPUT_FOLDER
        # Folder for machine learning efficiency scoring
        mleff_dir = os.path.join(output_dir, "mlefficiency") # this folder will save the scores
        cache_mleff_dir = os.path.join(mleff_dir, "cache") # this folder will save the outputs
        outpath = os.path.join(mleff_dir, "board.csv")
        os.makedirs(cache_mleff_dir, exist_ok=True)
        configure.OUTPUT_FOLDER = cache_mleff_dir

        # Folder for competition scoring
        score_comp_folder = os.path.join(output_dir, "comp")
        score_comp_cache_folder = os.path.join(score_comp_folder, "cache")
        os.makedirs(score_comp_cache_folder, exist_ok=True)
        score_comp_file = os.path.join(score_comp_folder, "score.csv")

        # train ml efficiency

        scores = sdgym.run(synthesizers=model, datasets=[dataset_name],cache_dir=mleff_dir, output_path=outpath, iterations=3)

        # train competition
        configure.OUTPUT_FOLDER = score_comp_cache_folder
        real_data, meta_json = check_data_exist(dataset=dataset_name)
        train = pd.read_csv(real_data)
        f = open(meta_json, "r")
        meta = json.load(f)
        f.close()
        categorical, ordinal, columns = get_columns(train, meta)
        train = train.values.astype('float32')
        net = model()
        synth = net.fit_sample_ret(train, categorical, ordinal)
        output_synth_file = os.path.join(score_comp_cache_folder, "output.csv")
        synth = pd.DataFrame(synth, columns=columns)
        synth = normalize_values(synth, meta)
        synth.to_csv(output_synth_file, index=False)

        output_score_file = os.path.join(score_comp_folder, "score.csv")
        score1, score2 = calculate_comp_score(dataset_name, output_synth_file)
        avg_score = float(score1 + score2) / 2
        f = open(score_comp_file, "w")
        f.write("score1, score2, avg\n")
        f.write("%f, %f, %f\n" % (score1, score2, avg_score))
        f.close()

        # calculate nmi values
        real_frame = pd.read_csv(real_data)
        synth_frame = pd.read_csv(output_synth_file)
        nmi_set = nmi_distance(real_frame, synth_frame, meta)
        print("Pairwise column nmi distance absolute: %f" % nmi_set[0])
        print("Pairwise column nmi square mean root distance: %f" % nmi_set[1])
        log_nmi_column = os.path.join(score_comp_folder, "column_nmi.csv")

        log_nmi_column_distance = os.path.join(score_comp_folder, "columns_pnmi_distance.csv")
        count = 0
        f = open(log_nmi_column_distance, "w")
        for i in range(len(columns)):
            for j in range(i + 1, len(columns), 1):
                f.write("%s, %s, %f, %f\n" % (columns[i], columns[j], nmi_set[2][count], nmi_set[3][count]))
                count += 1
        f.close()
    else:
        output_dir = configure.OUTPUT_FOLDER
        # Folder saving competion logs
        score_comp_folder = os.path.join(output_dir, "comp")
        score_comp_cache_folder = os.path.join(score_comp_folder, "cache")
        os.makedirs(score_comp_cache_folder, exist_ok=True)
        score_comp_file = os.path.join(score_comp_folder, "score.csv")
        configure.OUTPUT_FOLDER = score_comp_cache_folder

        # real files, ground truth for scoring
        real_data, meta_json = check_data_exist(dataset=dataset_name)
        train = pd.read_csv(real_data)
        f = open(meta_json, "r")
        meta = json.load(f)
        f.close()
        categorical, ordinal, columns = get_columns(train, meta)
        train = train.values.astype('float32')
        net = model()
        net.sample_checkpoints(columns)

        # folder for checkpoints output file
        checkpoints_folder = os.path.join(score_comp_cache_folder, "checkpoints")
        ckpoints_files = os.path.join(checkpoints_folder, "ckpt_files")
        score_checkpoints = os.path.join(checkpoints_folder, "score.csv")
        f = open(score_checkpoints, "w")
        f.write("epoch, score1, score2, avg\n")
        # process checkpoints nets
        for i in range(configure.EPOCHS):
            if (i + 1) >= 200 and (i + 1) % 10 == 0:
                epoch_output = "epoch_%d.csv" % i
                synth_file = os.path.join(ckpoints_files, epoch_output)
                synth = pd.read_csv(synth_file)
                synth = normalize_values(synth, meta)
                synth.to_csv(synth_file, index=False)
                score1, score2 = calculate_comp_score(dataset_name, synth_file)
                f.write("%d, %f, %f, %f\n" % (
                    i, score1, score2, (score1 + score2) / 2))
        f.close()

