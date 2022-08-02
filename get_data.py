import json
import pandas as pd


def get_evaluation_data(file_name):
    f = open(file_name)
    queries_json = json.load(f)
    f.close()
    return queries_json


def get_canonical_df(canonical_file_name):
    header_list = ['id', 'paragraph']
    df = pd.read_csv(canonical_file_name, names=header_list, sep='\t')
    df = df.iloc[1:]
    return df
