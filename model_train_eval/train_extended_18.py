"""
Fine-tune and save a multi-label classification model with Simple Transformers.

The script can be customized with the following parameters:
    --datapath: data dir
    --train_pkl: the file with the train data
    --eval_pkl: the file with the eval data
    --config: json file containing the model args
    --model_args: the name of the model args dict from `config`
    --model_type: type of the pre-trained model, e.g. bert, roberta, electra
    --modelpath: models dir
    --model_name: the pre-trained model, either from Hugging Face or locally stored
    --gold_col: column header of the gold labels
    --hf: pass this parameter if a model from Hugging Face is used

To change the default values of a parameter, pass it in the command line, e.g.:

$ python train_model.py --model_name pdelobelle/robbert-v2-dutch-base --hf
"""


import logging
import warnings
import torch
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
import ast

import sys
sys.path.insert(0, '..')

import os
os.environ['WANDB_MODE'] = 'offline'

def train(
    train_pkl,
    eval_pkl,
    model_args,
    model_type,
    model_name,
    gold_col,
    labels=["B1300 Energy level",
    "B140 Attention functions",
    "B152 Emotional functions",
    "B440 Respiration functions",
    "B455 Exercise tolerance functions",
    "B530 Weight maintenance functions",
    "D450 Walking",
    "D550 Eating",
    "D840-D859 Work and employment",
    "B280 Sensations of pain",
    "B134 Sleep functions",
    "D760 Family relationships",
    "B164 Higher-level cognitive functions",
    "D465 Moving around using equipment",
    "D410 Changing basic body position",
    "B230 Hearing functions",
    "D240 Handling stress and other psychological demands",
    "None"],
):
    """
    Fine-tune and save a multi-label classification model with Simple Transformers.

    Parameters
    ----------
    train_pkl: str
        path to pickled df with the training data, which must contain the columns 'text' and 'labels'; the labels are multi-hot lists (see column indices in `labels`), e.g. [1, 0, 0, 1, 0, 0, 0, 0, 1]
    eval_pkl: {None, str}
        path to pickled df for evaluation during training (optional)
    config_json: str
        path to a json file containing the model args
    args: str
        the name of the model args dict from `config_json` to use
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        the exact architecture and trained weights to use; this can be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model file
    labels: list
        list of column indices for the multi-hot labels

    Returns
    -------
    None
    """

    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn(' ====== CUDA device not available; running on a CPU! ====== ')

    # logging
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.WARNING)

    # load data
    train_data = pd.read_pickle(train_pkl)
    train_data = train_data.rename({gold_col:'labels'}, axis=1)
    train_df = train_data[['text','labels']].copy()
    train_df['labels'] = train_df['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    #print(" ====== train df: ======\n",train_df)

    eval_data = pd.read_pickle(eval_pkl)
    eval_data = eval_data.rename({gold_col:'labels'}, axis=1)
    eval_df = eval_data[['text','labels']].copy()
    eval_df['labels'] = eval_df['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    #print(" ====== eval df: ======\n",eval_df)

    # model args
    # with open(config_json, 'r') as f:
    #     config = json.load(f)
    # model_args = config[args]
    #model_args = model_args

    #print("model_args", model_args)

    # model
    model = MultiLabelClassificationModel(
        model_type,
        model_name,
        num_labels=len(labels),
        args=model_args,
        use_cuda=cuda_available,
        ignore_mismatched_sizes=True,
    )

    # train
    if model.args.evaluate_during_training:
        model.train_model(train_df, eval_df=eval_df)
    else:
        print(" ====== Training starts ====== ")
        model.train_model(train_df)


if __name__ == '__main__':

    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('--datapath', default='data_expr_july', help='must be listed as a key in /config.ini')
    # argparser.add_argument('--train_pkl', default='/otherdisk/data/cecilia_k/a-proof-zonmw/data/combine_jenia_gpt')
    # argparser.add_argument('--eval_pkl', default='/otherdisk/data/cecilia_k/a-proof-zonmw/data/dev_gpt.pkl', help='only used if `evaluate_during_training` is True in the model args in `config`')
    # argparser.add_argument('--config', default='config.json')
    # argparser.add_argument('--model_args', default='baseline_gpt_j')
    # argparser.add_argument('--model_type', default='roberta')
    # argparser.add_argument('--modelpath', default='/otherdisk/data/cecilia_k/a-proof-zonmw/models')
    # argparser.add_argument('--model_name', default='/mnt/data/A-Proof/data2/a-proof-zonmw/medroberta')
    # argparser.add_argument('--hf', dest='hugging_face', action='store_true')
    # argparser.set_defaults(hugging_face=False)
    # args = argparser.parse_args()

    train_pkl = 'train.pkl'
    eval_pkl = 'dev.pkl'
    model_type = 'roberta'
    model_name = 'models'
    gold_col = 'labels_18'

    # NOT USING EARLY STOPPING
    model_args = {
        "max_seq_length": 512,
        "output_dir": "medroberta_18cats",
        "best_model_dir": "./medroberta_18cats/best_model/",
        "tensorboard_dir": "./medroberta_18cats/runs/",
        "cache_dir": "./medroberta_18cats/cache_dir/",
        "dataloader_num_workers": 4,
        "process_count":12,
        "use_multiprocessing": False,
        "use_multiprocessing_for_evaluation": False,
        "silent": False,
        "manual_seed": 42,
        "num_train_epochs": 5,
        "learning_rate" : 4e-5,
        "train_batch_size": 8,
        "save_eval_checkpoints": False,
        "save_steps": -1,
        "overwrite_output_dir": True,
        "evaluate_during_training": True,
        "evaluate_during_training_steps": 1000,
        "save_steps": -1,
        "evaluate_during_training_verbose": True, # optional, prints eval results clearly
        }
        #"use_early_stopping": True,
        #"early_stopping_delta": 0.01,
        #"early_stopping_metric": "mcc",
        #"early_stopping_metric_minimize": False,
        # "early_stopping_patience": 5,
        # "evaluate_during_training": True,
        # "evaluate_during_training_steps": 1000,

    # if args.hugging_face:
    #     model_name = args.model_name


    train(
        train_pkl,
        eval_pkl,
        model_args,
        model_type,
        model_name,
        gold_col
    )
