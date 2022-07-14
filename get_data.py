import json
from read_data import *
import pandas as pd


def get_evaluation_data(file_name):
    f = open(file_name)
    queries_json = json.load(f)
    f.close()
    return queries_json


def get_ms_marco_data(file_name):
    return pd.read_csv(file_name, names=['id', 'paragraph'], sep='\t')


def get_car_collection(file_name):
    ids, paragraphs = [], []
    for p in iter_paragraphs(file_name):
        ids.append(p.para_id)
        paragraphs.append(p)
        print(p.para_id)
        print(p)

    car_df = pd.DataFrame(data=[ids, paragraphs], names=['id', 'paragraph'])
    return car_df
