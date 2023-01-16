from pygaggle.rerank.transformer import MonoBERT
from pyserini.search.lucene import LuceneSearcher
from simplet5 import SimpleT5
from transformers import pipeline

from get_data import get_evaluation_data
from runs.testing_metrics import testing_metrics
from runs.automatic import automatic
from runs.canonical import canonical
from runs.testing_retrieval_reranking import testing_retrieval_reranking
from constants import *

if __name__ == '__main__':
    print("starting main.py")
    evaluation_data_json = get_evaluation_data(EVALUATION_DATA_FILE_LOCATION)
    searcher = LuceneSearcher(SEARCHER_DATA_FILE_LOCATION)
    reranker = MonoBERT()
    our_T5 = SimpleT5()
    our_T5.load_model("t5", PERSONAL_T5_REWRITER_CONFIG_LOCATION, use_gpu=True)
    base_T5 = SimpleT5()
    base_T5.load_model("t5", BASE_T5_REWRITER_CONFIG_LOCATION, use_gpu=True)
    summarizeT5 = pipeline("summarization", model=T5_SUMMARIZER_CONFIG_LOCATION)

    print("loaded all models")
    testing_metrics(evaluation_data_json)
    testing_retrieval_reranking(evaluation_data_json, searcher, reranker)

    for data_usage_type in DATA_USAGE_TYPES:
        automatic(evaluation_data_json, searcher, reranker, our_T5, data_usage_type, summarizeT5, "personal")
        canonical(evaluation_data_json, searcher, reranker, our_T5, data_usage_type, summarizeT5, "personal")

        automatic(evaluation_data_json, searcher, reranker, base_T5, data_usage_type, summarizeT5, "base")
        canonical(evaluation_data_json, searcher, reranker, base_T5, data_usage_type, summarizeT5, "base")
