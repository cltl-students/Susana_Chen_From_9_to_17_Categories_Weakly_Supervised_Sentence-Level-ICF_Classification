import pandas as pd
import csv
import random

IN_CSV = "newcats_sentences_2023.csv"
OUT_CSV = "newcats_sentences_2023_shuf.csv"

df = pd.read_csv(IN_CSV, sep=';', engine='python', quoting=csv.QUOTE_NONE, encoding='utf-8')

sentences = df.to_dict('records')

random.shuffle(sentences)

pd.DataFrame(sentences).to_csv(OUT_CSV, sep=';', header=True, index=False, encoding='utf-8', quoting=csv.QUOTE_NONE)
