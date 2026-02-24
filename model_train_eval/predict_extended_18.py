"""
Apply a fine-tuned multi-label classification model to generate predictions.
The text is given in a pickled df and the predictions are generated per row and saved in a 'predictions' column.

The script can be customized with the following parameters:
    --datapath: data dir
    --data_pkl: the file with the text
    --model_type: type of the fine-tuned model, e.g. bert, roberta, electra
    --modelpath: models dir
    --model_name: the fine-tuned model, locally stored

To change the default values of a parameter, pass it in the command line, e.g.:

$ python predict.py --datapath data_expr_sept
python3 'predict_copy.py'
"""


import argparse
import warnings
import torch
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
from pathlib import Path
import numpy as np
from scipy.special import softmax
import sys
sys.path.insert(0, '.')
#from utils.config import PATHS
# from utils.timer import timer


# @timer
def predict_df(
    data_pkl,
    model_type,
    model_name,
):
    """
    Apply a fine-tuned multi-label classification model to generate predictions.
    The text is given in `data_pkl` and the predictions are generated per row and saved in a 'predictions' column.

    Parameters
    ----------
    data_pkl: str
        path to pickled df with the data, which must contain the column 'text'
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        path to a directory containing model file

    Returns
    -------
    None
    """

    # load data
    print("starting now")
    df = pd.read_pickle(data_pkl)

    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn('CUDA device not available; running on a CPU!')

    # load model
    model = MultiLabelClassificationModel(
        model_type,
        model_name,
        use_cuda=cuda_available,
    )

    # predict
    print("Generating predictions and confidence. This might take a while...")
    txt = df['text'].to_list()
    predictions, raw_outputs = model.predict(txt)
    probabilities = [*([*softmax(element)] for element in raw_outputs)]

    col = f"pred_{Path(model_name).stem}"
    confidence = f"confidence_{Path(model_name).stem}"
    df[col] = predictions
    df[confidence] = probabilities

    # pkl df
    df.to_pickle(data_pkl)
    print(f"A column with predictions and a column with the confidence score were added.\nThe updated df is saved: {data_pkl}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--datapath', default='data', help='must be listed as a key in /config.ini')
    argparser.add_argument('--data_pkl', default='combined_test_new_INS_fixed_FP.pkl')
    argparser.add_argument('--model_type', default='roberta')
    argparser.add_argument('--modelpath', default='medroberta_18cats')
    argparser.add_argument('--model_name', default='medroberta_18cats')
    args = argparser.parse_args()

    data_pkl = "./data/combined_test_new_INS_fixed_FP.pkl"
    model_name = "medroberta_18cats"
    model_type = 'roberta'

    predict_df(
        data_pkl,
        model_type,
        model_name,
    )

#all_data_10.pkl
#all_primary_10.pkl
