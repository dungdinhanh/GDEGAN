import sdgym

import argparse
import os


from models import configure

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
parser.add_argument('--model', default=0 ,type=int, help='index of model')
parser.add_argument('--dataset', default="adult", type=str ,help='benchmark dataset')
parser.add_argument('--epochs', default=300, type=int, help='Number of epochs for training')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
parser.add_argument('--batch', default=500, type=int, help='batch size')
parser.add_argument('--outputdir', default='output', type=str, help='output folder')
parser.add_argument('--latent', default=25, type=int, help='latent size for autoencoder')
parser.add_argument('--alpha', default=0.1, type=float, help='weight for reconstruction loss')
parser.add_argument('--test', action='store_true', help='train or load checkpoints')
parser.add_argument('--pac', default=10, type=int, help="define pac size")
parser.add_argument('--transformer', default=0, type=int, help="index for choosing transformer, 0 for normal, 1 for discretize low card integer")
parser.add_argument('--graph', default=0, type=int, help='mode for constructing graphs')
parser.add_argument('--iters', default=3, type=int, help='number of iterations')
# parser = argparse.ArgumentParser(description='Process some integers.')
args = parser.parse_args()
configure.EPOCHS = args.epochs
configure.BATCH = args.batch
configure.CUDA = args.cuda

configure.PAC = args.pac
configure.TRANSFORMER = configure.LIST_TRANSFORMER[args.transformer]

# must be imported after changing configure (import before causing errors)
from models.ctgan import CTGANSynthesizer
from models.daegan import DEGANSynthesizer
from models.graph_daegan import GraphCondDAEGAN
from models.graph_ctgan import GraphCondCTGAN
from models.daegan_res import DEGANResSynthesizer
from models.privbayes import PrivBayesSynth
from models.medgan import MedGAN
from models.vee import VEEGAN

models = [CTGANSynthesizer, DEGANSynthesizer, DEGANResSynthesizer, GraphCondCTGAN, GraphCondDAEGAN, PrivBayesSynth, MedGAN, VEEGAN]
from utils.utils import *
from utils.graph import *
import json
from utils.dataset_stat import *


if __name__ == '__main__':
    model = models[args.model]
    dataset_name = args.dataset
    configure.LATENT = args.latent
    alpha = configure.ALPHA = args.alpha
    model_temp = model()
    if args.model in [0, 3]:
        configure.OUTPUT_FOLDER = os.path.join(configure.OUTPUT_FOLDER, model_temp.__class__.__name__ + "pac%d"%args.pac, args.dataset)
    else:
        configure.OUTPUT_FOLDER = os.path.join(configure.OUTPUT_FOLDER, model_temp.__class__.__name__ , args.dataset)

    # Get folder mark
    if args.model in [1, 2, 4]:
        int_alpha, frac_alpha = str(alpha).split(".")[0], str(alpha).split(".")[1]
        alpha_mark = "%sp%s"%(int_alpha, frac_alpha)
        output_dir = os.path.join(configure.OUTPUT_FOLDER,
                                  "latent%d_alpha%s"%(configure.LATENT, alpha_mark))
    else:
        output_dir = configure.OUTPUT_FOLDER

    if dataset_name in ["grid", "gridr"]:
        configure.TRANSFORMER = configure.LIST_TRANSFORMER[0]
        # print(configure.TRANSFORMER)

    if args.model in [3, 4]:
        if dataset_name in ["firedept"]:
            graph_nx = get_graph_nan(dataset_name, 3, args.graph)
        else:
            graph_nx = get_graph(dataset_name, 3, args.graph)
        if graph_nx is not None:
            graph = nx.to_numpy_matrix(graph_nx).astype(int)
        else:
            graph = graph_nx
        configure.GRAPH = graph
        output_dir = os.path.join(output_dir, "graphmode%d"%args.graph)


    if not args.test:
        # This folder is the tuning folders
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

        scores = sdgym.run(synthesizers=model, datasets=[dataset_name],cache_dir=mleff_dir, output_path=outpath, iterations=args.iters)

        # train competition

        if dataset_name not in ["alarm", "child", "asia", "grid", "gridr"]:
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
    else:
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

