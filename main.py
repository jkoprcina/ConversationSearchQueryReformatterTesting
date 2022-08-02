from pygaggle.rerank.transformer import MonoBERT
from pyserini.search.lucene import LuceneSearch
from simplet5 import SimpleT5
import pandas as pd

from get_data import *
from testing_main import *
from canonical_testing_main import *


if __name__ == '__main__':
    EVALUATION_DATA_FILE_NAME = "data/2020_automatic_evaluation_topics_v1.json"
    CANONICAL_MS_MARCO_DATA_FILE_NAME = "data/canonical_ms_marco"
    CANONICAL_CAR_DATA_FILE_NAME = "data/canonical_car"

    evaluation_data_json = get_evaluation_data(EVALUATION_DATA_FILE_NAME)
    print("Evaluation json returned.")
    canonical_ms_marco_df = get_canonical_df(CANONICAL_MS_MARCO_DATA_FILE_NAME)
    print("Canonical MS Marco returned.")
    canonical_car_df = get_canonical_df(CANONICAL_CAR_DATA_FILE_NAME)
    print("Canonical car returned.")

    # Set the searcher/ranker that uses the index made out of car and msmarco datasets
    searcher = LuceneSearch('data/indexes/lucene-index-msmarco-passage')
    # Set the reranker to MonoBERT
    reranker = MonoBERT()

    # Set and test the query rewriter
    T5 = SimpleT5()
    T5.load_model("t5", "data/simplet5", use_gpu=True)
    test = "How much does it cost for someone to fix it? Why did garage door opener stop working? How do you know " \
           "when your garage door opener is going bad? "
    T5.predict(test)
    print("T5 ready and tested.")

    main_loop(evaluation_data_json, searcher, reranker, T5, "h2oloo")
    canonical_main_loop(canonical_ms_marco_df, canonical_car_df, evaluation_data_json, T5, "h2oloo")
